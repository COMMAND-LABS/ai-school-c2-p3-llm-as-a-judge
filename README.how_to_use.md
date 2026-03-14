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
python evaluate_kalygo_agent.py \
  --experiment-name "kalygo-asa-agent-with-openai-llm-gpt-4o-mini" \
  --dataset-name "kalygo-ai-school-kb-3-12-2026" \
  --max-examples 50 \
  --timeout-seconds 90
```

```sh - Experiment 2
python evaluate_kalygo_agent.py \
  --experiment-name "kalygo-asa-agent-with-openai-llm-gpt-5.4" \
  --dataset-name "kalygo-ai-school-kb-3-12-2026" \
  --max-examples 50 \
  --timeout-seconds 90
```
