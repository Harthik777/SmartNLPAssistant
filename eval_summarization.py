import os, re, time, glob, csv, argparse, numpy as np
from transformers import pipeline
from rouge_score import rouge_scorer

def chunk_text(text, max_tokens=700, max_chunks=40):
    sents = re.split(r'(?<=[.!?])\s+', text)
    chunks, cur, n = [], [], 0
    for s in sents:
        toks = s.split()
        if n + len(toks) > max_tokens:
            if cur: chunks.append(" ".join(cur))
            cur, n = [], 0
        cur.extend(toks); n += len(toks)
        if len(chunks) >= max_chunks: break
    if cur and len(chunks) < max_chunks:
        chunks.append(" ".join(cur))
    return [c for c in chunks if len(c.split()) > 20]

def summarize_doc(text, model_name="distilbart", two_pass=True):
    if model_name == "t5-small":
        summarizer = pipeline("summarization", model="t5-small", device=-1)
    else:
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
    chunks = chunk_text(text, max_tokens=700, max_chunks=40)
    if not chunks: return "", 0.0
    t0 = time.perf_counter()
    parts = []
    for c in chunks:
        out = summarizer(c, max_length=200, min_length=80, do_sample=False, truncation=True)
        parts.append(out[0]["summary_text"])
    if two_pass and len(parts) > 1:
        merged = " ".join(parts)
        out2 = summarizer(merged, max_length=260, min_length=140, do_sample=False, truncation=True)
        summary = out2[0]["summary_text"]
    else:
        summary = " ".join(parts)
    t1 = time.perf_counter()
    return summary, (t1 - t0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs_dir", default="data/summarization/docs")
    ap.add_argument("--refs_dir", default="data/summarization/refs")
    ap.add_argument("--model", choices=["distilbart","t5-small"], default="distilbart")
    ap.add_argument("--two_pass", action="store_true")
    ap.add_argument("--out_csv", default="sum_results.csv")
    args = ap.parse_args()

    scorer = rouge_scorer.RougeScorer(["rouge1","rougeL"], use_stemmer=True)
    doc_files = sorted(glob.glob(os.path.join(args.docs_dir, "*.txt")))
    rows, r1s, rLs, times = [], [], [], []

    for fdoc in doc_files:
        fname = os.path.basename(fdoc)
        fref = os.path.join(args.refs_dir, fname)
        if not os.path.exists(fref):
            print(f"[SKIP] missing reference for {fname}")
            continue
        doc = open(fdoc, encoding="utf-8", errors="ignore").read()
        ref = open(fref, encoding="utf-8", errors="ignore").read()

        hyp, secs = summarize_doc(doc, model_name=args.model, two_pass=args.two_pass)
        sc = scorer.score(ref, hyp)
        r1, rL = sc["rouge1"].fmeasure, sc["rougeL"].fmeasure

        r1s.append(r1); rLs.append(rL); times.append(secs)
        rows.append([fname, args.model, int(args.two_pass), f"{r1:.3f}", f"{rL:.3f}", f"{secs:.2f}"])
        print(f"{fname:10s}  R1={r1:.3f}  RL={rL:.3f}  time={secs:.2f}s")

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file","model","two_pass","rouge1","rougeL","time_sec"])
        w.writerows(rows)

    if r1s:
        print("\n=== Averages ===")
        print(f"ROUGE-1={np.mean(r1s):.3f}  ROUGE-L={np.mean(rLs):.3f}  Avg time={np.mean(times):.2f}s")

if __name__ == "__main__":
    main()
