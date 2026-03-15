"""Configuration and argument parsing for the runner."""

import argparse
import os

from dotenv import load_dotenv

from llm_judge.metrics import AVAILABLE_METRICS, DEFAULT_EVALUATORS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Kalygo agent against a Q&A CSV with LangSmith.")
    parser.add_argument(
        "--dataset-file",
        "--csv-path",
        dest="dataset_file",
        default=os.getenv("DATASET_FILE", "data/ai_school_kb_3-12-2026.csv"),
        help=(
            "Path to the local Q&A CSV file. "
            "Use --dataset-name for the LangSmith dataset identifier."
        ),
    )
    parser.add_argument("--agent-id", default=os.getenv("KALYGO_AGENT_ID"), help="Kalygo agent ID")
    parser.add_argument(
        "--kalygo-completion-api-url",
        "--api-url",
        dest="kalygo_completion_api_url",
        default=os.getenv("KALYGO_COMPLETION_API_URL", "https://completion.kalygo.io"),
        help="Base URL for Kalygo API (completion + agent config)",
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
        "--kalygo-api-timeout-seconds",
        "--timeout-seconds",
        dest="kalygo_api_timeout_seconds",
        type=int,
        default=int(os.getenv("KALYGO_API_TIMEOUT_SECONDS", os.getenv("KALYGO_TIMEOUT_SECONDS", "120"))),
        help="HTTP timeout for each Kalygo API completion request",
    )
    parser.add_argument(
        "--judge-model",
        default=os.getenv("JUDGE_MODEL", "gpt-4o-mini"),
        help="Model used by the LLM judge evaluator",
    )
    parser.add_argument(
        "--evaluators",
        default=os.getenv("EVALUATORS", DEFAULT_EVALUATORS),
        help=(
            "Comma-separated evaluators to run. "
            f"Available: {', '.join(AVAILABLE_METRICS)}"
        ),
    )
    parser.add_argument(
        "--judge-timeout-seconds",
        type=int,
        default=int(os.getenv("JUDGE_TIMEOUT_SECONDS", "60")),
        help="HTTP timeout for judge model API calls",
    )
    return parser.parse_args()


def load_config() -> argparse.Namespace:
    load_dotenv()
    return parse_args()
