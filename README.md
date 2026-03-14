# Kalygo Agent Evaluation with LangSmith

This project evaluates Q&A pairs in `data/ai_school_kb_3-12-2026.csv` against a Kalygo agent completion endpoint and logs experiment runs to LangSmith.

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

3. Fill values in `.env`:
   - `KALYGO_AI_API_KEY`
   - `KALYGO_AGENT_ID`
   - `LANGSMITH_API_KEY`
   - Optional: `LANGSMITH_DATASET_NAME` (reuse one dataset across runs)

## Run

```bash
python evaluate_kalygo_agent.py
```

Optional flags:

```bash
python evaluate_kalygo_agent.py --experiment-name "ai-school-eval-march-12" --max-examples 50 --timeout-seconds 90
```

If `LANGSMITH_DATASET_NAME` is not set, the script creates a timestamped dataset for each run.
If `LANGSMITH_DATASET_NAME` is set, the script replaces that dataset's examples each run so `--max-examples` is applied exactly.

Each request now sends `sessionId` as a UUID (`uuid4`) to satisfy the Kalygo endpoint requirements.

## What it evaluates

- Exact match (normalized text)
- Token-level F1 overlap
- Reference answer contained in model output
