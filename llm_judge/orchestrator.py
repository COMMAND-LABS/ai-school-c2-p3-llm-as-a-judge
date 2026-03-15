"""Top-level orchestration for LLM-as-a-judge evaluation runs."""

import json
import os
import uuid
from typing import Any

from langsmith import Client
from langsmith.evaluation import evaluate

from llm_judge.config import load_config
from llm_judge.dataset import ensure_dataset_with_examples, load_examples, resolve_dataset_name
from llm_judge.evaluator_selection import build_selected_evaluators
from llm_judge.kalygo_client import call_kalygo_completion, fetch_kalygo_agent_config


def _mask_secret(value: str | None) -> str:
    if not value:
        return "<not set>"
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


def _print_run_overview(
    *,
    args: Any,
    dataset_name: str,
    experiment_name: str,
    judge_base_url: str,
    kalygo_api_key: str | None,
    judge_api_key: str | None,
    requested_evaluators: list[str],
    enabled_evaluators: list[str],
    skipped_evaluators: list[str],
) -> None:
    print("")
    print("=" * 72)
    print("RUN CONFIGURATION OVERVIEW")
    print("=" * 72)
    print(f"Dataset file         : {args.dataset_file}")
    print(f"LangSmith dataset    : {dataset_name}")
    print(f"Experiment name      : {experiment_name}")
    print(f"Agent ID             : {args.agent_id}")
    print(f"Kalygo API URL       : {args.kalygo_completion_api_url}")
    print(f"Kalygo API key       : {_mask_secret(kalygo_api_key)}")
    print(f"Max examples         : {args.max_examples if args.max_examples is not None else 'all'}")
    print(f"Kalygo API timeout   : {args.kalygo_api_timeout_seconds}")
    print(f"Kalygo API retries   : {args.kalygo_api_retries}")
    print(f"Judge model          : {args.judge_model}")
    print(f"Judge API base URL   : {judge_base_url}")
    print(f"Judge API key        : {_mask_secret(judge_api_key)}")
    print(f"Judge timeout (sec)  : {args.judge_timeout_seconds}")
    print(f"Requested evaluators : {', '.join(requested_evaluators)}")
    print(f"Enabled evaluators   : {', '.join(enabled_evaluators)}")
    print(
        "Skipped evaluators   : "
        f"{', '.join(skipped_evaluators) if skipped_evaluators else '<none>'}"
    )
    print("=" * 72)
    print("")


def _print_agent_config_snapshot(agent_config_payload: dict[str, Any] | None, fetch_error: str | None) -> None:
    print("")
    print("=" * 72)
    print("KALYGO AGENT CONFIGURATION SNAPSHOT")
    print("=" * 72)
    if fetch_error:
        print(f"Unable to fetch agent config: {fetch_error}")
    elif agent_config_payload is None:
        print("Agent config was not requested.")
    else:
        print(json.dumps(agent_config_payload, indent=2, sort_keys=True))
    print("=" * 72)
    print("")


def main() -> None:
    # 1) Load CLI/env configuration for this evaluation run.
    args = load_config()

    # 2) Validate required runtime settings and gather optional provider config.
    if not args.agent_id:
        raise ValueError("Missing agent id. Set KALYGO_AGENT_ID in .env or pass --agent-id.")

    api_key = os.getenv("KALYGO_API_KEY", os.getenv("KALYGO_AI_API_KEY"))
    if not os.getenv("LANGSMITH_API_KEY"):
        raise ValueError("Missing LANGSMITH_API_KEY in .env (required for LangSmith evaluation).")
    judge_api_key = os.getenv("OPENAI_API_KEY")
    judge_base_url = os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1")

    # 3) Load examples from the local dataset file.
    examples = load_examples(args.dataset_file, max_examples=args.max_examples)
    if not examples:
        raise ValueError(f"No valid examples found in {args.dataset_file}")

    # 4) Resolve naming used in LangSmith for this run.
    experiment_name = (args.experiment_name or "").strip() or args.experiment_prefix
    dataset_name = resolve_dataset_name(args.dataset_name)

    # 5) Ensure LangSmith dataset exists and contains exactly these examples.
    client = Client()
    ensure_dataset_with_examples(client=client, dataset_name=dataset_name, examples=examples)

    # Pull the current Kalygo agent config for run-time reference in logs.
    agent_config_payload: dict[str, Any] | None = None
    agent_config_error: str | None = None
    try:
        agent_config_payload = fetch_kalygo_agent_config(
            api_url=args.kalygo_completion_api_url,
            api_key=api_key,
            agent_id=str(args.agent_id),
            kalygo_api_timeout_seconds=args.kalygo_api_timeout_seconds,
        )
    except Exception as exc:
        agent_config_error = (
            f"{exc}. If completion works but this fails, verify both "
            "KALYGO_COMPLETION_API_URL and KALYGO_API_KEY permissions/scopes "
            "for GET /api/agents/{agent_id}."
        )

    # 6) Emit a concise run summary before evaluation starts.
    print(f"Loaded {len(examples)} examples from {args.dataset_file}")
    print(f"Synced dataset: {dataset_name}")

    # 7) Define the system-under-test callable passed into LangSmith `evaluate`.
    def target(inputs: dict[str, str]) -> dict[str, str]:
        question = inputs["question"]
        session_id = str(uuid.uuid4())
        answer = call_kalygo_completion(
            api_url=args.kalygo_completion_api_url,
            api_key=api_key,
            agent_id=args.agent_id,
            session_id=session_id,
            prompt=question,
            kalygo_api_timeout_seconds=args.kalygo_api_timeout_seconds,
            kalygo_api_retries=args.kalygo_api_retries,
        )
        return {"answer": answer}

    # 8) Build the selected evaluator functions from CLI/env configuration.
    evaluators, requested, enabled, skipped = build_selected_evaluators(
        evaluator_csv=args.evaluators,
        judge_api_key=judge_api_key,
        judge_model=args.judge_model,
        judge_base_url=judge_base_url,
        judge_timeout_seconds=args.judge_timeout_seconds,
    )

    _print_run_overview(
        args=args,
        dataset_name=dataset_name,
        experiment_name=experiment_name,
        judge_base_url=judge_base_url,
        kalygo_api_key=api_key,
        judge_api_key=judge_api_key,
        requested_evaluators=requested,
        enabled_evaluators=enabled,
        skipped_evaluators=skipped,
    )
    _print_agent_config_snapshot(agent_config_payload=agent_config_payload, fetch_error=agent_config_error)
    print("Running LangSmith evaluation...")

    # 9) Execute evaluation and attach useful metadata for experiment analysis.
    experiment_results = evaluate(
        target,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix=experiment_name,
        client=client,
        metadata={
            "agent_id": args.agent_id,
            "dataset_file": args.dataset_file,
            "kalygo_completion_api_url": args.kalygo_completion_api_url,
            "dataset_name": dataset_name,
            "experiment_name_config": experiment_name,
            "judge_model": args.judge_model,
            "requested_evaluators": requested,
            "enabled_evaluators": enabled,
        },
    )

    # 10) Print post-run links/details for quick navigation in LangSmith.
    experiment_name_output = getattr(experiment_results, "experiment_name", None)
    experiment_url = getattr(experiment_results, "experiment_url", None)
    print("Evaluation finished.")
    if experiment_name_output:
        print(f"Experiment: {experiment_name_output}")
    if experiment_url:
        print(f"View in LangSmith: {experiment_url}")
