import argparse
import os

from dotenv import load_dotenv
from langsmith import Client
from langsmith import schemas as ls_schemas
from langsmith.evaluation import evaluate

from llm_judge.metrics import build_llm_judge_evaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload an LLM-as-a-judge evaluator run to LangSmith."
    )
    parser.add_argument(
        "--dataset-name",
        default="llm-judge-evaluator-demo-dataset",
        help="LangSmith dataset name to create/use for this demo.",
    )
    parser.add_argument(
        "--experiment-name",
        default="llm-judge-evaluator-demo",
        help="Experiment prefix shown in LangSmith.",
    )
    parser.add_argument(
        "--judge-model",
        default=os.getenv("JUDGE_MODEL", "gpt-4o-mini"),
        help="Model name used by the LLM-as-a-judge evaluator.",
    )
    parser.add_argument(
        "--judge-timeout-seconds",
        type=int,
        default=int(os.getenv("JUDGE_TIMEOUT_SECONDS", "60")),
        help="Timeout for judge model API calls.",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="If dataset exists, replace all examples before running.",
    )
    return parser.parse_args()


def ensure_dataset(client: Client, dataset_name: str, replace_existing: bool) -> None:
    examples = [
        {
            "inputs": {"question": "What is the capital of France?"},
            "outputs": {"answer": "Paris"},
        },
        {
            "inputs": {"question": "What planet do humans live on?"},
            "outputs": {"answer": "Earth"},
        },
    ]

    created = False
    if not client.has_dataset(dataset_name=dataset_name):
        client.create_dataset(
            dataset_name=dataset_name,
            description="Tiny dataset for LLM judge evaluator demo",
            data_type=ls_schemas.DataType.kv,
        )
        created = True
    elif replace_existing:
        ids = [example.id for example in client.list_examples(dataset_name=dataset_name)]
        if ids:
            for i in range(0, len(ids), 100):
                client.delete_examples(ids[i : i + 100])

    if created or replace_existing:
        client.create_examples(dataset_name=dataset_name, examples=examples)
        return

    current = list(client.list_examples(dataset_name=dataset_name, limit=1))
    if not current:
        client.create_examples(dataset_name=dataset_name, examples=examples)


def target(inputs: dict[str, str]) -> dict[str, str]:
    question = inputs["question"].lower()
    if "capital of france" in question:
        return {"answer": "Paris"}
    if "planet do humans live on" in question:
        return {"answer": "Earth"}
    return {"answer": "I am not sure."}


def main() -> None:
    load_dotenv()
    args = parse_args()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is required to run LLM-as-a-judge demo.")

    judge_base_url = os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1")
    client = Client()
    ensure_dataset(client, args.dataset_name, args.replace_existing)

    llm_judge = build_llm_judge_evaluator(
        model_name=args.judge_model,
        api_key=openai_api_key,
        base_url=judge_base_url,
        timeout_seconds=args.judge_timeout_seconds,
    )

    results = evaluate(
        target,
        data=args.dataset_name,
        evaluators=[llm_judge],
        experiment_prefix=args.experiment_name,
        client=client,
        metadata={"judge_model": args.judge_model},
    )

    print(f"Uploaded LLM judge evaluator run for dataset: {args.dataset_name}")
    url = getattr(results, "experiment_url", None)
    if url:
        print(f"View in LangSmith: {url}")


if __name__ == "__main__":
    main()
