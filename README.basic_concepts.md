# TLDR

##

```sh
python upload_langsmith_dataset.py \
  --dataset-file data/ai_school_kb_shorter_3_15_2026.csv \
  --dataset-name ai_school_kb_shorter_3_15_2026 \
  --replace-existing
```

##

```sh
python upload_simple_evaluator_demo.py \
 --dataset-name simple-evaluator-demo-dataset \
 --experiment-name simple-evaluator-demo \
 --replace-existing
```

##

```sh
python upload_llm_judge_evaluator_demo.py \
 --dataset-name llm-judge-evaluator-demo-dataset \
 --experiment-name llm-judge-evaluator-demo \
 --judge-model gpt-4o-mini \
 --replace-existing
```
