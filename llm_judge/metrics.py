"""Metric definitions for LangSmith evaluations."""

import json
import re
from typing import Any

import requests

LLM_JUDGE_METRIC_KEY = "llm_judge_score"
DEFAULT_EVALUATORS = "exact_match,token_f1,contains_reference,llm_judge_score"
AVAILABLE_METRICS = (
    "exact_match",
    "token_f1",
    "contains_reference",
    LLM_JUDGE_METRIC_KEY,
)


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


def _parse_float_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


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


def build_llm_judge_evaluator(
    *,
    model_name: str,
    api_key: str,
    base_url: str,
    timeout_seconds: int,
) -> Any:
    def llm_judge_evaluator(run: Any, example: Any) -> dict[str, Any]:
        question = (example.inputs or {}).get("question", "")
        expected = (example.outputs or {}).get("answer", "")
        predicted = (run.outputs or {}).get("answer", "")

        judge_prompt = (
            "You are an impartial evaluator.\n"
            "Score whether the predicted answer is correct for the question, using the reference answer.\n"
            "Return JSON only in this exact schema: "
            '{"score": <number between 0 and 1>, "reasoning": "<short reason>"}.\n\n'
            f"Question: {question}\n"
            f"Reference answer: {expected}\n"
            f"Predicted answer: {predicted}\n"
        )

        response = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json={
                "model": model_name,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": "You are a strict grading assistant."},
                    {"role": "user", "content": judge_prompt},
                ],
            },
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        content = payload.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        parsed = json.loads(content)
        return {
            "key": LLM_JUDGE_METRIC_KEY,
            "score": _parse_float_score(parsed.get("score")),
            "comment": str(parsed.get("reasoning", "")),
        }

    return llm_judge_evaluator
