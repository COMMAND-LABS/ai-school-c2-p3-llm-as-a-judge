# LLM-as-a-Judge Evaluation with LangSmith

This project demonstrates how to implement LLM-as-a-Judge evaluation on the LangSmith platform.

## Project structure

Core logic is split into small modules under `llm_judge/`:

- `llm_judge/config.py` - CLI + environment configuration
- `llm_judge/kalygo_client.py` - API call + stream parsing
- `llm_judge/dataset.py` - CSV loading + LangSmith dataset sync
- `llm_judge/evaluator_selection.py` - evaluator selection/validation
- `llm_judge/metrics.py` - metric definitions and LLM-judge evaluator
- `llm_judge/orchestrator.py` - top-level run orchestration

`run_llm_as_a_judge.py` is now a thin entrypoint that calls the orchestrator.

## Setup

1. Create and activate a virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Fill in `.env` value:
   - This is going to be specific to exactly what you are testing
   - peep the `.env.example` file for reference
   - Set `JUDGE_MODEL` to choose which LLM acts as the judge (for example `gpt-4o-mini` or `gpt-4.1-mini`)
   - Set `EVALUATORS` to control which metrics are active
   - Set `KALYGO_COMPLETION_API_URL` to the Kalygo API base (completion + agent config, e.g. `https://completion.kalygo.io`)
   - Set `KALYGO_API_KEY` for authenticated completion and agent-config calls
   - Set `KALYGO_API_TIMEOUT_SECONDS` for Kalygo completion request timeout
   - Optional: set `KALYGO_API_RETRIES` for transient timeout/connection retries

## Run

```bash
python run_llm_as_a_judge.py --dataset-file data/ai_school_kb_3-12-2026.csv
```

Override the judge model for a single run:

```bash
python run_llm_as_a_judge.py --dataset-file data/ai_school_kb_3-12-2026.csv --judge-model gpt-4.1-mini
```

Use only selected evaluators for a run:

```bash
python run_llm_as_a_judge.py --evaluators exact_match,token_f1
```

## What it evaluates

- Metric definitions live in `llm_judge/metrics.py` for easy review.
- Each metric includes inline documentation explaining:
  - what the metric measures
  - score interpretation
  - where it is useful
- Included metrics:
  - `exact_match`
  - `token_f1`
  - `contains_reference`
  - `llm_judge_score` (enabled when `OPENAI_API_KEY` is set)
- `--evaluators` / `EVALUATORS` lets you choose any subset of those metrics.
