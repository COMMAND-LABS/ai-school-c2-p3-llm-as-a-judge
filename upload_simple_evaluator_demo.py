import argparse

from dotenv import load_dotenv
from langsmith import Client
from langsmith import schemas as ls_schemas
from langsmith.evaluation import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a simple custom evaluator result to LangSmith."
    )
    parser.add_argument(
        "--dataset-name",
        default="simple-evaluator-demo-dataset",
        help="LangSmith dataset name to create/use for this demo.",
    )
    parser.add_argument(
        "--experiment-name",
        default="simple-evaluator-demo",
        help="Experiment prefix shown in LangSmith.",
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
            "inputs": {"color": "blue"},
            "outputs": {"must_include": "blue"},
        },
        {
            "inputs": {"color": "green"},
            "outputs": {"must_include": "green"},
        },
    ]

    if not client.has_dataset(dataset_name=dataset_name):
        client.create_dataset(
            dataset_name=dataset_name,
            description="Minimal dataset for simple custom evaluator demo",
            data_type=ls_schemas.DataType.kv,
        )
    elif replace_existing:
        ids = [example.id for example in client.list_examples(dataset_name=dataset_name)]
        if ids:
            for i in range(0, len(ids), 100):
                client.delete_examples(ids[i : i + 100])

    if replace_existing or not client.has_dataset(dataset_name=dataset_name):
        client.create_examples(dataset_name=dataset_name, examples=examples)
        return

    # If dataset exists and caller didn't request replace, only seed if empty.
    current = list(client.list_examples(dataset_name=dataset_name, limit=1))
    if not current:
        client.create_examples(dataset_name=dataset_name, examples=examples)


def target(inputs: dict[str, str]) -> dict[str, str]:
    # Deliberately simple model/system behavior for demo purposes.
    color = inputs["color"]
    return {"answer": f"My favorite color is {color}."}


def keyword_inclusion_evaluator(run, example) -> dict:
    predicted = (run.outputs or {}).get("answer", "").lower()
    keyword = (example.outputs or {}).get("must_include", "").lower()
    score = 1.0 if keyword and keyword in predicted else 0.0
    return {
        "key": "keyword_inclusion",
        "score": score,
        "comment": f"Checks whether '{keyword}' appears in answer.",
    }


def main() -> None:
    load_dotenv()
    args = parse_args()
    client = Client()

    ensure_dataset(client, args.dataset_name, args.replace_existing)
    results = evaluate(
        target,
        data=args.dataset_name,
        evaluators=[keyword_inclusion_evaluator],
        experiment_prefix=args.experiment_name,
        client=client,
    )

    print(f"Uploaded simple evaluator run for dataset: {args.dataset_name}")
    url = getattr(results, "experiment_url", None)
    if url:
        print(f"View in LangSmith: {url}")


if __name__ == "__main__":
    main()
