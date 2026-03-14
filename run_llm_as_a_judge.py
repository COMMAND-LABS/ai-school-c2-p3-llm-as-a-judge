import argparse
import csv
import datetime as dt
import json
import os
import re
import uuid
from typing import Any
from urllib.parse import quote

import requests
from dotenv import load_dotenv
from langsmith import Client
from langsmith import schemas as ls_schemas
from langsmith.evaluation import evaluate


def _normalize_text(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"\s+", " ", value)
    return re.sub(r"[^a-z0-9\s]", "", value)


def _token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _normalize_text(prediction).split()
    ref_tokens = _normalize_text(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts: dict[str, int] = {}
    ref_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    overlap = 0
    for token, count in pred_counts.items():
        overlap += min(count, ref_counts.get(token, 0))

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def _extract_text_from_payload(payload: Any) -> str:
    if isinstance(payload, str):
        return payload

    if isinstance(payload, dict):
        for key in ("completion", "output", "text", "delta", "token", "answer", "response"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        if "message" in payload:
            return _extract_text_from_payload(payload["message"])
        if "choices" in payload and isinstance(payload["choices"], list):
            chunks = [_extract_text_from_payload(choice) for choice in payload["choices"]]
            return "".join(chunk for chunk in chunks if chunk)

    if isinstance(payload, list):
        chunks = [_extract_text_from_payload(item) for item in payload]
        return "".join(chunk for chunk in chunks if chunk)

    return ""


def _parse_sse_data_line(data_value: str) -> str:
    if data_value == "[DONE]":
        return ""
    try:
        parsed = json.loads(data_value)
    except json.JSONDecodeError:
        return data_value
    return _extract_text_from_payload(parsed)


def _iter_json_objects(payload_text: str) -> list[dict[str, Any]]:
    """Parse one or more concatenated JSON objects from a payload string."""
    decoder = json.JSONDecoder()
    idx = 0
    length = len(payload_text)
    objects: list[dict[str, Any]] = []

    while idx < length:
        while idx < length and payload_text[idx].isspace():
            idx += 1
        if idx >= length:
            break
        try:
            parsed, end_idx = decoder.raw_decode(payload_text, idx)
        except json.JSONDecodeError:
            break
        if isinstance(parsed, dict):
            objects.append(parsed)
        idx = end_idx

    return objects


def call_kalygo_completion(
    *,
    api_url: str,
    api_key: str | None,
    agent_id: str,
    session_id: str,
    prompt: str,
    timeout_seconds: int = 120,
) -> str:
    url = f"{api_url.rstrip('/')}/api/agents/{quote(agent_id, safe='')}/completion"
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response = requests.post(
        url,
        headers=headers,
        json={"sessionId": session_id, "prompt": prompt},
        stream=True,
        timeout=timeout_seconds,
    )

    if not response.ok:
        details = response.text
        raise RuntimeError(f"Kalygo API error {response.status_code}: {details}")

    stream_chunks: list[str] = []
    final_answer: str | None = None
    current_event = "message"

    def _handle_payload(payload_text: str, event_hint: str) -> str | None:
        nonlocal final_answer
        if payload_text == "[DONE]":
            return "done"

        parsed_objects = _iter_json_objects(payload_text)
        for parsed in parsed_objects:
            payload_event = parsed.get("event")
            event_name = payload_event if isinstance(payload_event, str) and payload_event else event_hint

            if event_name == "error":
                error_data = parsed.get("data")
                if isinstance(error_data, dict):
                    error_message = error_data.get("message") or error_data.get("error")
                else:
                    error_message = parsed.get("message") or parsed.get("error") or error_data
                raise RuntimeError(f"Kalygo stream error: {error_message or 'unknown error'}")

            if event_name == "on_chain_end":
                chain_data = parsed.get("data")
                if isinstance(chain_data, str):
                    final_answer = chain_data.strip()
                elif chain_data is not None:
                    final_answer = _extract_text_from_payload(chain_data).strip()
                continue

            # Keep a fallback transcript if on_chain_end is missing.
            if event_name == "on_chat_model_stream":
                stream_piece = parsed.get("data")
                if isinstance(stream_piece, str) and stream_piece:
                    stream_chunks.append(stream_piece)
                continue

        if parsed_objects:
            return None

        if event_hint == "error":
            raise RuntimeError(f"Kalygo stream error: {payload_text}")

        # Ignore non-JSON control/noise lines so they do not pollute evaluation outputs.
        return None

    for line in response.iter_lines(decode_unicode=True):
        if line is None:
            continue
        stripped = line.strip()
        if not stripped:
            current_event = "message"
            continue
        if stripped.startswith("event:"):
            current_event = stripped[6:].strip() or "message"
            continue
        if stripped.startswith(("id:", "retry:")):
            continue

        # Support both SSE (`data: ...`) and raw JSON-lines event streams.
        payload_text = stripped[5:].strip() if stripped.startswith("data:") else stripped
        outcome = _handle_payload(payload_text, current_event)
        if outcome == "done":
            break

    if final_answer is not None:
        return final_answer
    return "".join(stream_chunks).strip()


def load_examples(csv_path: str, max_examples: int | None = None) -> list[dict[str, dict[str, str]]]:
    examples: list[dict[str, dict[str, str]]] = []
    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            question = (row.get("q") or "").strip()
            answer = (row.get("a") or "").strip()
            if not question or not answer:
                continue
            examples.append({"inputs": {"question": question}, "outputs": {"answer": answer}})
            if max_examples is not None and len(examples) >= max_examples:
                break
    return examples


def _ensure_dataset_with_examples(
    *,
    client: Client,
    dataset_name: str,
    examples: list[dict[str, dict[str, str]]],
) -> None:
    if not client.has_dataset(dataset_name=dataset_name):
        client.create_dataset(
            dataset_name=dataset_name,
            description="Kalygo AI School Q&A evaluation dataset",
            data_type=ls_schemas.DataType.kv,
        )
    else:
        # Keep dataset rows exactly aligned with this run's CSV slice (e.g. --max-examples).
        existing_ids = [example.id for example in client.list_examples(dataset_name=dataset_name)]
        if existing_ids:
            chunk_size = 100
            for idx in range(0, len(existing_ids), chunk_size):
                client.delete_examples(existing_ids[idx : idx + chunk_size])
    client.create_examples(dataset_name=dataset_name, examples=examples)


def exact_match_evaluator(run: Any, example: Any) -> dict[str, Any]:
    predicted = (run.outputs or {}).get("answer", "")
    expected = (example.outputs or {}).get("answer", "")
    score = 1.0 if _normalize_text(predicted) == _normalize_text(expected) else 0.0
    return {"key": "exact_match", "score": score}


def token_f1_evaluator(run: Any, example: Any) -> dict[str, Any]:
    predicted = (run.outputs or {}).get("answer", "")
    expected = (example.outputs or {}).get("answer", "")
    return {"key": "token_f1", "score": _token_f1(predicted, expected)}


def substring_contains_evaluator(run: Any, example: Any) -> dict[str, Any]:
    predicted = _normalize_text((run.outputs or {}).get("answer", ""))
    expected = _normalize_text((example.outputs or {}).get("answer", ""))
    score = 1.0 if expected and expected in predicted else 0.0
    return {"key": "contains_reference", "score": score}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Kalygo agent against a Q&A CSV with LangSmith.")
    parser.add_argument(
        "--csv-path",
        default="data/ai_school_kb_3-12-2026.csv",
        help="Path to the Q&A CSV. Defaults to data/ai_school_kb_3-12-2026.csv",
    )
    parser.add_argument("--agent-id", default=os.getenv("KALYGO_AGENT_ID"), help="Kalygo agent ID")
    parser.add_argument(
        "--api-url",
        default=os.getenv("KALYGO_AI_API_URL", "https://completion.kalygo.io"),
        help="Kalygo AI API URL",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Experiment name for this run. If omitted, --experiment-prefix is used.",
    )
    parser.add_argument(
        "--experiment-prefix",
        default="kalygo-default-prefix",
        help="Prefix for LangSmith experiment name",
    )
    parser.add_argument(
        "--dataset-name",
        default=os.getenv("LANGSMITH_DATASET_NAME"),
        help="LangSmith dataset name. If omitted, a timestamped dataset name is created.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on number of CSV examples to evaluate",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=int(os.getenv("KALYGO_TIMEOUT_SECONDS", "120")),
        help="HTTP timeout for each completion request",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    if not args.agent_id:
        raise ValueError("Missing agent id. Set KALYGO_AGENT_ID in .env or pass --agent-id.")

    api_key = os.getenv("KALYGO_AI_API_KEY")
    if not os.getenv("LANGSMITH_API_KEY"):
        raise ValueError("Missing LANGSMITH_API_KEY in .env (required for LangSmith evaluation).")

    examples = load_examples(args.csv_path, max_examples=args.max_examples)
    if not examples:
        raise ValueError(f"No valid examples found in {args.csv_path}")

    experiment_name = (args.experiment_name or "").strip() or args.experiment_prefix

    if args.dataset_name:
        dataset_name = args.dataset_name
    else:
        timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")
        dataset_name = f"kalygo-ai-school-kb-{timestamp}"

    client = Client()
    _ensure_dataset_with_examples(client=client, dataset_name=dataset_name, examples=examples)

    print(f"Loaded {len(examples)} examples from {args.csv_path}")
    print(f"Synced dataset: {dataset_name}")
    print(f"Experiment name config: {experiment_name}")
    print("Running LangSmith evaluation...")

    def target(inputs: dict[str, str]) -> dict[str, str]:
        question = inputs["question"]
        session_id = str(uuid.uuid4())
        answer = call_kalygo_completion(
            api_url=args.api_url,
            api_key=api_key,
            agent_id=args.agent_id,
            session_id=session_id,
            prompt=question,
            timeout_seconds=args.timeout_seconds,
        )
        return {"answer": answer}

    experiment_results = evaluate(
        target,
        data=dataset_name,
        evaluators=[exact_match_evaluator, token_f1_evaluator, substring_contains_evaluator],
        experiment_prefix=experiment_name,
        client=client,
        metadata={
            "agent_id": args.agent_id,
            "csv_path": args.csv_path,
            "api_url": args.api_url,
            "dataset_name": dataset_name,
            "experiment_name_config": experiment_name,
        },
    )

    experiment_name = getattr(experiment_results, "experiment_name", None)
    experiment_url = getattr(experiment_results, "experiment_url", None)
    print("Evaluation finished.")
    if experiment_name:
        print(f"Experiment: {experiment_name}")
    if experiment_url:
        print(f"View in LangSmith: {experiment_url}")


if __name__ == "__main__":
    main()
