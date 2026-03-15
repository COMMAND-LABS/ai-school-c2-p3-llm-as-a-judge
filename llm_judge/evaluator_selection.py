"""Evaluator selection and validation."""

from typing import Any

from llm_judge.metrics import (
    AVAILABLE_METRICS,
    LLM_JUDGE_METRIC_KEY,
    build_llm_judge_evaluator,
    exact_match_evaluator,
    substring_contains_evaluator,
    token_f1_evaluator,
)


def build_selected_evaluators(
    *,
    evaluator_csv: str,
    judge_api_key: str | None,
    judge_model: str,
    judge_base_url: str,
    judge_timeout_seconds: int,
) -> tuple[list[Any], list[str], list[str], list[str]]:
    static_evaluator_registry: dict[str, Any] = {
        "exact_match": exact_match_evaluator,
        "token_f1": token_f1_evaluator,
        "contains_reference": substring_contains_evaluator,
    }

    requested_evaluators = [item.strip() for item in evaluator_csv.split(",") if item.strip()]
    unknown_evaluators = [name for name in requested_evaluators if name not in AVAILABLE_METRICS]
    if unknown_evaluators:
        raise ValueError(
            "Unknown evaluators requested: "
            f"{', '.join(unknown_evaluators)}. "
            f"Available: {', '.join(AVAILABLE_METRICS)}"
        )

    evaluators: list[Any] = []
    enabled_evaluator_names: list[str] = []
    skipped_evaluator_names: list[str] = []

    for evaluator_name in requested_evaluators:
        if evaluator_name in static_evaluator_registry:
            evaluators.append(static_evaluator_registry[evaluator_name])
            enabled_evaluator_names.append(evaluator_name)
            continue

        if evaluator_name == LLM_JUDGE_METRIC_KEY:
            if not judge_api_key:
                skipped_evaluator_names.append(f"{LLM_JUDGE_METRIC_KEY} (OPENAI_API_KEY not set)")
                continue
            evaluators.append(
                build_llm_judge_evaluator(
                    model_name=judge_model,
                    api_key=judge_api_key,
                    base_url=judge_base_url,
                    timeout_seconds=judge_timeout_seconds,
                )
            )
            enabled_evaluator_names.append(LLM_JUDGE_METRIC_KEY)

    if not evaluators:
        raise ValueError("No evaluators enabled. Check --evaluators and environment variables.")

    return evaluators, requested_evaluators, enabled_evaluator_names, skipped_evaluator_names
