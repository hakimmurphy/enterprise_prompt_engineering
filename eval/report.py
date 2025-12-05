import pandas as pd

def summarize(df: pd.DataFrame) -> dict:
    out = {
        "n": len(df),
        "acc": round(df["acc"].mean(), 4),
        "latency_s_avg": round(df["latency"].mean(), 4),
        "cost_total_usd": round(df["cost"].sum(), 4),
    }
    return out
