"""Dataset file loading and LangSmith dataset sync helpers."""

import csv
import datetime as dt
from typing import Any

from langsmith import Client
from langsmith import schemas as ls_schemas


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


def resolve_dataset_name(dataset_name: str | None) -> str:
    if dataset_name:
        return dataset_name
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")
    return f"kalygo-ai-school-kb-{timestamp}"


def ensure_dataset_with_examples(
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
        existing_ids = [example.id for example in client.list_examples(dataset_name=dataset_name)]
        if existing_ids:
            chunk_size = 100
            for idx in range(0, len(existing_ids), chunk_size):
                client.delete_examples(existing_ids[idx : idx + chunk_size])
    client.create_examples(dataset_name=dataset_name, examples=examples)
