# eval/judge.py  (LLM-as-judge + spot audits)
def rubric_judge(pred, gold):
    # simple rubric: exact match = 1, else 0
    return 1.0 if pred == gold else 0.0

def spot_audit(sample_text, model_rationale):
    # placeholder: store a few examples for manual review
    return {"text": sample_text[:120], "rationale": model_rationale[:200]}
