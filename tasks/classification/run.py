

# tasks/classification/run.py
import os, json, time, yaml, pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from eval.metrics import sanitize_label
from eval.report import summarize

load_dotenv()  # loads .env into os.environ

HERE = Path(__file__).parent

# --- Gemini call ---
import google.generativeai as genai
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# optional: tweak generation config (temperature, etc.)
GEN_CFG = {"temperature": 0.2, "top_p": 0.95, "top_k": 40}
MODEL_NAME = "gemini-2.5-flash" 
model = genai.GenerativeModel(MODEL_NAME, generation_config=GEN_CFG)

def gemini_llm(prompt_text: str) -> tuple[str, float, float]:
    start = time.time()
    try:
        resp = model.generate_content(prompt_text)
        text = (resp.text or "").strip()
    except Exception as e:
        text = f"ERROR: {e}"
    latency = time.time() - start
    cost = 0.0  # optional: leave 0 or compute from usage if you track it
    return text, latency, cost

def load_yaml(fp):
    with open(fp, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    ds_path = HERE / "dataset.jsonl"
    cfg = load_yaml(HERE / "prompts.yaml")
    rows = [json.loads(l) for l in open(ds_path, "r", encoding="utf-8")]

    os.makedirs(HERE / "results", exist_ok=True)

    system = cfg["system"]
    user_tpl = cfg["user_template"]

    # Open output log file
    output_log = HERE / "results" / "output.log"
    summary_log = HERE / "results" / "summary.log"
    all_summaries = []
    with open(output_log, "w", encoding="utf-8") as log:
        for var in cfg["variants"]:
            name = var["name"]
            fewshot = var.get("fewshot", [])

            preds, golds, lats, costs = [], [], [], []

            for r in rows:
                # build the prompt (system + few-shot + user content)
                few = "\n".join([f'Example:\n{fs["text"]}\nLabel: {fs["label"]}\n' for fs in fewshot])
                prompt = f"{system}\n{few}\n" + user_tpl.format(text=r["text"])

                text, latency, cost = gemini_llm(prompt)
                log.write(f"RAW OUTPUT for '{r['text']}': {text}\n")
                print(f"RAW OUTPUT for '{r['text']}': {text}")  # Debug raw model output
                pred = sanitize_label(text)

                preds.append(pred); golds.append(r["label"])
                lats.append(latency); costs.append(cost)

            df = pd.DataFrame({"pred": preds, "gold": golds, "latency": lats, "cost": costs})
            df["acc"] = (df["pred"] == df["gold"]).astype(int)

            out_csv = HERE / "results" / f"{name}.csv"
            df.to_csv(out_csv, index=False)
            log.write(f"[{name}] saved -> {out_csv}\n")
            print(f"[{name}] saved -> {out_csv}")
            summary = summarize(df)
            all_summaries.append(f"{name}: {summary}")
            log.write(f"Summary: {summary}\n")
            print("Summary:", summary)

    # Save all summaries to summary.log
    with open(summary_log, "w", encoding="utf-8") as s_log:
        for line in all_summaries:
            s_log.write(line + "\n")

if __name__ == "__main__":
    main()
