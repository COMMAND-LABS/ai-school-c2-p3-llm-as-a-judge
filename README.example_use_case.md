# TLDR

Overview of how to use this project

## STEPS

- You need something to be able to test
- Choose an LLM (or sufficiently competent AI system) that will act as your "judge"
  - ie:
- Vibe code a script that uploads a dataset (stored in the data folder) to LangSmith and evaluates the performance of some system (ie: an AI Agent on the Kalygo platform)
- Cleanly and Clearly label each dataset and experiment name to more easily interpret your results
- Compare your results in the LangSmith dashboard for extracting and applying the insights from your experiments

## EXAMPLE

```sh - Experiment 1
python run_llm_as_a_judge.py \
  --experiment-name "asa-agent-with-gpt-4o-mini" \
  --dataset-file "data/ai-school-kb-3-12-2026.csv" \
  --dataset-name "kalygo-ai-school-kb-3-12-2026" \
  --evaluators "llm_judge_score" \
  --judge-model "gpt-4o-mini" \
  --max-examples 10 \
  --kalygo-api-timeout-seconds 90 \
  --agent-id 39
```

```sh - Experiment 2
python run_llm_as_a_judge.py \
  --experiment-name "asa-agent-with-gpt-5.4" \
  --dataset-file "data/ai-school-kb-3-12-2026.csv" \
  --dataset-name "kalygo-ai-school-kb-3-12-2026" \
  --evaluators "llm_judge_score" \
  --judge-model "gpt-4o-mini" \
  --max-examples 20 \
  --kalygo-api-timeout-seconds 90 \
  --agent-id 40
```

```sh - Experiment 3
python run_llm_as_a_judge.py \
  --experiment-name "asa-agent-with-opus-4.6" \
  --dataset-file "data/ai-school-kb-3-12-2026.csv" \
  --dataset-name "kalygo-ai-school-kb-3-12-2026" \
  --evaluators "llm_judge_score" \
  --judge-model "gpt-4o-mini" \
  --max-examples 20 \
  --kalygo-api-timeout-seconds 90 \
  --agent-id 37
```

```sh - Experiment 4
python run_llm_as_a_judge.py \
  --experiment-name "asa-agent-with-gemini-3.1-pro" \
  --dataset-file "data/ai-school-kb-3-12-2026.csv" \
  --dataset-name "kalygo-ai-school-kb-3-12-2026" \
  --evaluators "llm_judge_score" \
  --judge-model "gpt-4o-mini" \
  --max-examples 20 \
  --kalygo-api-timeout-seconds 90 \
  --agent-id 35
```
