# Prompt Engineering Portfolio

Small, reproducible examples of **prompt design + evaluation**:
- Tasks: classification (sentiment) — more to come (retrieval, code-refactor)
- Variants: zero-shot vs few-shot with clear, measured **ACC / latency / cost**
- Ablations: show exactly what changed and why it helped

## How to run
```bash
pip install -r requirements.txt
python tasks/classification/run.py
python ablations/compare.py
