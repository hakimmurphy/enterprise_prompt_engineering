# eval/metrics.py
import re
from rouge_score import rouge_scorer
import sacrebleu

def sanitize_label(text: str) -> str:
    m = re.search(r'(POSITIVE|NEGATIVE)', text.upper())
    return m.group(1) if m else "UNKNOWN"

def exact_match(pred: str, gold: str) -> int:
    return int(pred.strip().upper() == gold.strip().upper())

def bleu(preds, refs):
    return sacrebleu.corpus_bleu(preds, [refs]).score

def rougeL(preds, refs):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = [scorer.score(r, p)['rougeL'].fmeasure for p, r in zip(preds, refs)]
    return sum(scores)/len(scores) if scores else 0.0

