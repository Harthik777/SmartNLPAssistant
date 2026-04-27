# app.py — AptX (CPU-only, Explainable, No OCR/Multilingual, No Vector DB)
# Features:
# - Summarization (DistilBART default, T5-small fallback), optional two-pass
# - QA with pure in-memory retrieval: BM25 -> MiniLM cosine re-rank
# - Explainability: citations table (BM25/Cosine/Fused), answer span highlight,
#   top influential sentences, keywords, confidence gating
# - Status timings and download files

import os, re, time, json, tempfile, concurrent.futures
import numpy as np
import gradio as gr
import fitz  # PyMuPDF
import torch, transformers
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from rank_bm25 import BM25Okapi
from rouge_score import rouge_scorer

# -----------------------------
# Logging + CPU tuning
# -----------------------------
transformers.logging.set_verbosity_error()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.set_num_threads(max(1, min(torch.get_num_threads(), os.cpu_count() or 1)))

# -----------------------------
# Models (CPU-only)
# -----------------------------
summarizer_distilbart = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
summarizer_t5 = pipeline("summarization", model="t5-small", device=-1)
qa_distilbert = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad", device=-1)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
kw_model = KeyBERT("all-MiniLM-L6-v2")

bm25_cache = {}
chunks_cache = {}

# -----------------------------
# Utils
# -----------------------------
TOKEN_RE = re.compile(r"\w+")

def tok(s: str):
    return TOKEN_RE.findall(s.lower())

def _t():
    return time.perf_counter()

def extract_text(path, max_chars=400_000):
    spath = str(path)
    if spath.lower().endswith(".pdf"):
        with fitz.open(spath) as doc:
            raw = "\n".join([p.get_text("text") for p in doc])
        return raw[:max_chars]
    else:
        with open(spath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()[:max_chars]

def chunk_text(text, max_tokens=700, max_chunks=40):
    sents = re.split(r'(?<=[.!?])\s+', text)
    chunks, cur, n = [], [], 0
    for s in sents:
        toks = s.split()
        if n + len(toks) > max_tokens:
            if cur:
                chunks.append(" ".join(cur))
            cur, n = [], 0
        cur.extend(toks); n += len(toks)
        if len(chunks) >= max_chunks:
            break
    if cur and len(chunks) < max_chunks:
        chunks.append(" ".join(cur))
    return [c for c in chunks if len(c.split()) > 20]

def _minmax(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    x_min, x_max = float(np.min(x)), float(np.max(x))
    if x_max == x_min:
        return np.full_like(x, 0.5)
    return (x - x_min) / (x_max - x_min)

def get_doc_id(text):
    return str(hash(text))

def ensure_bm25(chunks, doc_id):
    if doc_id not in bm25_cache:
        bm25_cache[doc_id] = BM25Okapi([tok(c) for c in chunks])
        chunks_cache[doc_id] = chunks
    return bm25_cache[doc_id]

# -----------------------------
# CSS-styled HTML renderers (fixed readability)
# -----------------------------
def highlight_answer_html(context, start, end):
    """Return HTML with answer span highlighted and surrounding text clearly visible."""
    start = max(0, start); end = max(start, end)
    def esc(s): return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

    before = esc(context[:start])
    middle = esc(context[start:end])
    after = esc(context[end:])

    return (
        "<div id='aptx-ctx'>"
        "<style>"
        "#aptx-ctx{background:#ffffff!important;color:#0f172a!important;"
        "font-family:'Inter','Segoe UI',system-ui,Roboto,Arial;"
        "line-height:1.7;border:1px solid #cbd5e1;border-radius:12px;padding:12px;}"
        "#aptx-ctx span,#aptx-ctx p{color:#0f172a!important;}"
        "#aptx-ctx mark{background:#dbeafe!important;color:#1e3a8a!important;"
        "padding:2px 4px;border-radius:4px;font-weight:600;}"
        "</style>"
        f"<span>{before}</span><mark>{middle}</mark><span>{after}</span>"
        "</div>"
    )

def passages_table_html(passages, metas, max_chars=420):
    rows = []
    for p, m in zip(passages, metas):
        txt = p if len(p) <= max_chars else (p[:max_chars] + "…")
        rows.append(
            "<tr>"
            f"<td><code>{m['chunk_index']}</code></td>"
            f"<td>{txt}</td>"
            f"<td class='num'>{m['bm25_norm']:.2f}</td>"
            f"<td class='num'>{m['cosine_norm']:.2f}</td>"
            f"<td class='num strong'>{m['fused']:.2f}</td>"
            "</tr>"
        )
    head = (
        "<thead><tr>"
        "<th>Chunk</th><th>Passage (truncated)</th>"
        "<th>BM25</th><th>Cosine</th><th>Fused</th>"
        "</tr></thead>"
    )
    body = "<tbody>" + "".join(rows) + "</tbody>"
    return (
        "<div id='aptx-table'>"
        "<style>"
        "#aptx-table{background:#ffffff!important;color:#0f172a!important;"
        "font-family:'Inter','Segoe UI',system-ui,Roboto,Arial;"
        "border:1px solid #cbd5e1;border-radius:12px;padding:10px;}"
        "#aptx-table table{width:100%;border-collapse:collapse;background:#ffffff!important;}"
        "#aptx-table th,#aptx-table td{border-bottom:1px solid #e5e7eb!important;"
        "padding:10px;vertical-align:top;color:#0f172a!important;line-height:1.6;}"
        "#aptx-table th{background:#e2e8f0!important;color:#0c4a6e!important;font-weight:700;text-align:left;}"
        "#aptx-table tbody tr:nth-child(even) td{background:#f8fafc!important;}"
        "#aptx-table td.num{text-align:right;}"
        "#aptx-table td.strong{font-weight:800;color:#1e3a8a!important;}"
        "#aptx-table code{background:#e2e8f0;color:#0f172a;padding:1px 4px;border-radius:4px;}"
        "</style>"
        "<div style='overflow:auto'><table>"
        + head + body +
        "</table></div></div>"
    )

# -----------------------------
# Summarization flow
# -----------------------------
def summarize_document(files, model_choice, enable_two_pass, max_tokens, max_chunks):
    t0 = _t()
    paths = [f if isinstance(f, str) else f.name for f in files]
    texts = [extract_text(p) for p in paths]
    doc_text = "\n".join(texts)
    chunks = chunk_text(doc_text, max_tokens=int(max_tokens), max_chunks=int(max_chunks))
    if not chunks:
        return "No readable content.", "", "", "", None, None, "Idle..."

    t1 = _t()
    summarizer = summarizer_t5 if model_choice == "T5-small" else summarizer_distilbart

    def _sum(c):
        out = summarizer(c, max_length=200, min_length=80, do_sample=False, truncation=True)
        return out[0]["summary_text"]

    summaries = []
    max_workers = min(4, (os.cpu_count() or 2))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for s in ex.map(_sum, chunks):
            summaries.append(s)

    if enable_two_pass and len(summaries) > 1:
        merged = " ".join(summaries)
        final_summary = summarizer(merged, max_length=260, min_length=140, do_sample=False, truncation=True)[0]["summary_text"]
    else:
        final_summary = " ".join(summaries)
    t2 = _t()

    doc_sentences = re.split(r'(?<=[.!?])\s+', doc_text)
    if len(doc_sentences) > 0:
        summary_emb = embedder.encode([final_summary], convert_to_numpy=True, normalize_embeddings=True)
        sent_embs = embedder.encode(doc_sentences, convert_to_numpy=True, normalize_embeddings=True)
        sim = np.dot(sent_embs, summary_emb.T).flatten()
        top_sentences = [doc_sentences[i] for i in np.argsort(sim)[-5:][::-1]]
    else:
        top_sentences = []

    keywords = [kw[0] for kw in kw_model.extract_keywords(final_summary, keyphrase_ngram_range=(1,2),
                                                          stop_words="english", top_n=12)]

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    score_list = [scorer.score(c, final_summary) for c in chunks]
    avg_r1 = float(np.mean([s["rouge1"].fmeasure for s in score_list])) if score_list else 0.0
    avg_rL = float(np.mean([s["rougeL"].fmeasure for s in score_list])) if score_list else 0.0
    rouge_text = f"Avg ROUGE-1: {avg_r1:.2f}, Avg ROUGE-L: {avg_rL:.2f}"

    fsum = tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8", suffix=".txt")
    fsum.write(final_summary); fsum.close()
    frep = tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8", suffix=".txt")
    frep.write("=== Summary ===\n"+final_summary+"\n\n")
    frep.write("=== ROUGE (proxy) ===\n"+rouge_text+"\n\n")
    frep.write("=== Top Influential Sentences ===\n"+"\n".join(top_sentences)+"\n\n")
    frep.write("=== Keywords ===\n"+", ".join(keywords)+"\n")
    frep.close()

    status = f"Extract+Chunk {(t1-t0):.2f}s | Summarize {(t2-t1):.2f}s"
    return final_summary, ", ".join(keywords), rouge_text, "\n".join(top_sentences), fsum.name, frep.name, status

# -----------------------------
# Retrieval + QA flow
# -----------------------------
def retrieve_passages(query, chunks, doc_id, top_k=3, fuse_w=0.6, pool_factor=5, pool_min=20):
    bm25 = ensure_bm25(chunks, doc_id)
    bm25_scores = bm25.get_scores(tok(query))

    N = min(max(pool_factor * top_k, pool_min), len(chunks))
    cand_idx = np.argsort(bm25_scores)[-N:][::-1]
    candidates = [chunks[i] for i in cand_idx]

    q_vec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    cand_vecs = embedder.encode(candidates, convert_to_numpy=True, normalize_embeddings=True)
    cos = cand_vecs @ q_vec

    bm25_cand = np.array([bm25_scores[i] for i in cand_idx])
    bm25_norm = _minmax(bm25_cand)
    cos_norm = _minmax(cos)

    w = float(fuse_w)
    fused = (1 - w) * bm25_norm + w * cos_norm

    order = np.argsort(fused)[-top_k:][::-1]
    top_texts = [candidates[i] for i in order]
    top_meta = []
    for i in order:
        top_meta.append({
            "chunk_index": int(cand_idx[i]),
            "bm25_norm": float(bm25_norm[i]),
            "cosine_norm": float(cos_norm[i]),
            "fused": float(fused[i]),
        })
    return top_texts, top_meta

def ask_question(files, question, fuse_w, top_k, conf_threshold, max_tokens, max_chunks):
    t0 = _t()
    paths = [f if isinstance(f, str) else f.name for f in files]
    texts = [extract_text(p) for p in paths]
    doc_text = "\n".join(texts)
    chunks = chunk_text(doc_text, max_tokens=int(max_tokens), max_chunks=int(max_chunks))
    if not chunks:
        return "No readable content.", "", "", None, "", "Idle..."

    doc_id = get_doc_id(doc_text)
    passages, metas = retrieve_passages(question, chunks, doc_id, top_k=int(top_k), fuse_w=float(fuse_w))
    t1 = _t()

    preds = []
    for p in passages:
        out = qa_distilbert(question=question, context=p)
        preds.append({
            "answer": out.get("answer",""),
            "score": float(out.get("score", 0.0)),
            "start": int(out.get("start", -1)),
            "end": int(out.get("end", -1)),
            "context": p
        })

    if not preds:
        return "No answer found.", "", passages_table_html(passages, metas), None, "", f"Q&A: Retrieve {(t1-t0):.2f}s | Reader 0.00s"

    best = max(preds, key=lambda x: x["score"])
    best_score = best["score"]
    best_answer = (best["answer"] or "").strip()
    abstain = (best_score < float(conf_threshold)) or (best_answer == "") or (best_answer in ["[CLS]"])

    if abstain:
        answer_text = "Insufficient evidence in the provided documents."
        best_html = None
    else:
        answer_text = best_answer
        best_html = highlight_answer_html(best["context"], best["start"], best["end"]) if best["start"] >= 0 else None

    ctx_keywords = [kw[0] for kw in kw_model.extract_keywords(best["context"], keyphrase_ngram_range=(1,2),
                                                              stop_words="english", top_n=10)] if best["context"] else []
    why_html = passages_table_html(passages, metas)

    ftxt = tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8", suffix=".txt")
    ftxt.write(answer_text); ftxt.close()
    fjson = tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8", suffix=".json")
    json.dump({"question": question, "best_answer": best_answer, "passages": metas}, fjson, ensure_ascii=False, indent=2)
    fjson.close()

    t2 = _t()
    status = f"Q&A: Retrieve {(t1 - t0):.2f}s | Reader {(t2 - t1):.2f}s"
    return answer_text, ", ".join(ctx_keywords), why_html, ftxt.name, fjson.name, best_html or None, status

# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(theme=gr.themes.Soft(primary_hue="sky", neutral_hue="slate")) as demo:
    gr.Markdown("## AptX — Efficient, Explainable Local NLP (CPU-only)")

    processing_status = gr.Textbox(label="Status", value="Idle...", interactive=False)

    with gr.Tab("Summarization"):
        files = gr.File(file_types=[".pdf", ".txt"], file_count="multiple", type="filepath", label="Upload PDF/TXT")
        with gr.Row():
            model_choice = gr.Radio(["DistilBART", "T5-small"], value="DistilBART", label="Summarizer")
            two_pass = gr.Checkbox(value=True, label="Two-Pass (better coherence)")
        with gr.Row():
            max_tokens = gr.Slider(300, 900, value=700, step=50, label="Chunk size (tokens)")
            max_chunks = gr.Slider(5, 60, value=40, step=1, label="Max chunks")
        summary_out = gr.Textbox(lines=14, label="Summary")
        keywords_out = gr.Textbox(label="Top Keywords")
        rouge_text = gr.Textbox(label="ROUGE (coverage proxy)")
        top_sents_out = gr.Textbox(label="Top Influential Sentences")
        download_summary = gr.File(label="Download Summary (.txt)")
        download_report = gr.File(label="Download Report (.txt)")
        summarize_btn = gr.Button("Generate Summary")
        summarize_btn.click(
            summarize_document,
            inputs=[files, model_choice, two_pass, max_tokens, max_chunks],
            outputs=[summary_out, keywords_out, rouge_text, top_sents_out, download_summary, download_report, processing_status]
        )

    with gr.Tab("Question Answering"):
        q_files = gr.File(file_types=[".pdf", ".txt"], file_count="multiple", type="filepath", label="Upload PDF/TXT")
        question = gr.Textbox(label="Your question")
        with gr.Row():
            fuse_w = gr.Slider(0.0, 1.0, value=0.6, step=0.05, label="Fusion weight (BM25 vs Cosine)")
            top_k = gr.Slider(1, 10, value=5, step=1, label="Top-K passages")
            conf_threshold = gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="Confidence threshold (abstain if below)")
        with gr.Row():
            q_max_tokens = gr.Slider(300, 900, value=700, step=50, label="Chunk size")
            q_max_chunks = gr.Slider(5, 60, value=40, step=1, label="Max chunks")
        answer_out = gr.Textbox(lines=6, label="Answer")
        ctx_keywords_out = gr.Textbox(label="Context Keywords")
        why_html = gr.HTML(label="Why this answer? (Top passages & scores)")
        download_qa = gr.File(label="Download Answer (.txt)")
        download_trace = gr.File(label="Download Trace (.json)")
        answer_html = gr.HTML(label="Answer in Context (highlighted)")
        ask_btn = gr.Button("Answer")
        ask_btn.click(
            ask_question,
            inputs=[q_files, question, fuse_w, top_k, conf_threshold, q_max_tokens, q_max_chunks],
            outputs=[answer_out, ctx_keywords_out, why_html, download_qa, download_trace, answer_html, processing_status]
        )

# Local run
if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, show_api=False)
