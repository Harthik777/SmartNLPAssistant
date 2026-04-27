import json, time, csv, argparse, numpy as np
from transformers import pipeline

def em_f1(pred, truth):
    p = pred.strip().lower()
    t = truth.strip().lower()
    em = int(p == t)
    pt, tt = p.split(), t.split()
    if not pt or not tt:
        return em, 0.0
    common = set(pt) & set(tt)
    f1 = 2 * len(common) / (len(pt) + len(tt)) if common else 0.0
    return em, f1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa_path", default="data/qa/qa.jsonl")
    ap.add_argument("--out_csv", default="qa_results.csv")
    args = ap.parse_args()

    qa = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad", device=-1)

    rows, ems, f1s, times = [], [], [], []
    with open(args.qa_path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip(): continue
            ex = json.loads(line)
            ctx = open(ex["context_path"], encoding="utf-8", errors="ignore").read()
            q = ex["question"]; gold = ex["answers"][0]
            t0 = time.perf_counter()
            out = qa(question=q, context=ctx)
            t1 = time.perf_counter()
            pred = out.get("answer", "")
            em, f1 = em_f1(pred, gold)
            ems.append(em); f1s.append(f1); times.append(t1 - t0)
            rows.append([i, ex["context_path"], q, gold, pred, f"{out.get('score',0):.3f}", f"{t1 - t0:.3f}", em, f"{f1:.3f}"])
            print(f"#{i:02d} EM={em} F1={f1:.3f} t={t1 - t0:.2f}s pred='{pred}'")

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id","context","question","gold","pred","confidence","time_sec","EM","F1"])
        w.writerows(rows)

    if ems:
        print("\n=== Averages ===")
        print(f"EM={np.mean(ems):.3f}  F1={np.mean(f1s):.3f}  Avg time={np.mean(times):.2f}s")

if __name__ == "__main__":
    main()
