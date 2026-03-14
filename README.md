# LLM-as-a-Judge Evaluation with LangSmith

This project demonstrates how to implement LLM-as-a-Judge evaluation on the LangSmith platform.

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

## Run

```bash
python run_llm_as_a_judge.py
```

## What it evaluates

- This is going to be specific to exactly what you are testing
  - Exact match (normalized text)
  - Token-level F1 overlap
  - Reference answer contained in model output
