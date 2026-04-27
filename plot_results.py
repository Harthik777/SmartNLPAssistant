# eval_summarization_extractive.py
# Evaluate a simple extractive summarization baseline and print metrics in console.

import os, glob, time, numpy as np, re, math
from collections import Counter
from rouge_score import rouge_scorer

# -----------------------------
# Extractive Summarizer
# -----------------------------
def _sent_tokenize(text):
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text)]
    return [s for s in sents if len(s.split()) >= 5]

def _word_tokenize(text):
    return re.findall(r"[A-Za-z0-9']+", text.lower())

def _tfidf_over_sents(sents):
    docs = [Counter(_word_tokenize(s)) for s in sents]
    df = Counter()
    for d in docs:
        for w in d:
            df[w] += 1
    N = len(docs)
    scores = []
    for d in docs:
        score = 0.0
        for w, tf in d.items():
            idf = math.log((1 + N) / (1 + df[w])) + 1.0
            score += tf * idf
        scores.append(score)
    return scores

def extractive_summary(text, max_words=120, max_sentences=6):
    sents = _sent_tokenize(text)
    if not sents:
        return ""
    scores = _tfidf_over_sents(sents)
    k = min(max_sentences, len(sents))
    top_idx = sorted(sorted(range(len(sents)), key=lambda i: scores[i], reverse=True)[:k])
    out, count = [], 0
    for i in top_idx:
        w = len(_word_tokenize(sents[i]))
        if count + w <= max_words or not out:
            out.append(sents[i])
            count += w
        else:
            break
    return " ".join(out)

# -----------------------------
# Evaluation Loop
# -----------------------------
DOCS_DIR = "data/summarization/docs"
REFS_DIR = "data/summarization/refs"

scorer = rouge_scorer.RougeScorer(["rouge1","rougeL"], use_stemmer=True)

r1s, rLs, times = [], [], []

print("Evaluating Extractive Baseline...\n")

for fdoc in sorted(glob.glob(os.path.join(DOCS_DIR, "*.txt"))):
    fname = os.path.basename(fdoc)
    fref = os.path.join(REFS_DIR, fname)
    if not os.path.exists(fref):
        print(f"[SKIP] Missing reference for {fname}")
        continue

    with open(fdoc, encoding="utf-8", errors="ignore") as f:
        doc = f.read()
    with open(fref, encoding="utf-8", errors="ignore") as f:
        ref = f.read()

    t0 = time.perf_counter()
    hyp = extractive_summary(doc)
    dt = time.perf_counter() - t0

    sc = scorer.score(ref, hyp)
    r1, rL = sc["rouge1"].fmeasure, sc["rougeL"].fmeasure
    r1s.append(r1)
    rLs.append(rL)
    times.append(dt)

    print(f"{fname:10s}  ROUGE-1={r1:.3f}  ROUGE-L={rL:.3f}  Time={dt:.2f}s")

# -----------------------------
# Summary Stats
# -----------------------------
if r1s:
    print("\n=== Averages (Extractive) ===")
    print(f"ROUGE-1={np.mean(r1s):.3f}  ROUGE-L={np.mean(rLs):.3f}  Avg Time={np.mean(times):.2f}s")
else:
    print("\nNo valid documents found in:", DOCS_DIR)
