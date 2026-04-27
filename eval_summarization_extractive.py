# eval_qa_multi.py
# Evaluate multiple QA reader models on data/qa/qa.jsonl and PRINT results (no files).

import json, time, argparse, numpy as np
from transformers import pipeline

def em_f1(pred, truth):
    """Simple EM/F1 consistent with your earlier script."""
    p = (pred or "").strip().lower()
    t = (truth or "").strip().lower()
    em = int(p == t)
    pt, tt = p.split(), t.split()
    if not pt or not tt:
        return em, 0.0
    common = set(pt) & set(tt)
    f1 = 2 * len(common) / (len(pt) + len(tt)) if common else 0.0
    return em, f1

def run_model(model_name, qa_path):
    qa = pipeline("question-answering", model=model_name, device=-1)
    ems, f1s, times = [], [], []
    n = 0

    print(f"\n=== {model_name} ===")
    with open(qa_path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            with open(ex["context_path"], encoding="utf-8", errors="ignore") as cf:
                ctx = cf.read()
            q = ex["question"]
            gold = ex["answers"][0]

            t0 = time.perf_counter()
            out = qa(question=q, context=ctx)
            dt = time.perf_counter() - t0

            pred = out.get("answer", "")
            em, f1 = em_f1(pred, gold)
            ems.append(em); f1s.append(f1); times.append(dt); n += 1

            # compact per-item line
            print(f"#{i:02d} EM={em} F1={f1:.3f} t={dt:.3f}s  pred='{pred}'")

    avg_em = float(np.mean(ems)) if ems else 0.0
    avg_f1 = float(np.mean(f1s)) if f1s else 0.0
    avg_t  = float(np.mean(times)) if times else 0.0
    print(f"\n-> {model_name} Averages: EM={avg_em:.3f}  F1={avg_f1:.3f}  Avg time={avg_t:.3f}s  (n={n})")
    return avg_em, avg_f1, avg_t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa_path", default="data/qa/qa.jsonl")
    ap.add_argument("--models", nargs="+", default=[
        "distilbert-base-uncased-distilled-squad",
        "csarron/mobilebert-uncased-squad-v2",
        "deepset/roberta-base-squad2"
    ])
    args = ap.parse_args()

    print("Device set to use cpu")
    results = []
    for m in args.models:
        em, f1, t = run_model(m, args.qa_path)
        results.append((m, em, f1, t))

    # neat summary table
    print("\n================ Summary (QA) ================")
    print(f"{'Model':37s} {'EM':>7s} {'F1':>7s} {'Avg t (s)':>10s}")
    for m, em, f1, t in results:
        print(f"{m:37s} {em:7.3f} {f1:7.3f} {t:10.3f}")
    print("==============================================")

if __name__ == "__main__":
    main()
