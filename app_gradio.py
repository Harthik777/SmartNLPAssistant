# app_gradio.py
import gradio as gr
import fitz, re, tempfile
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from rank_bm25 import BM25Okapi
from wordcloud import WordCloud
from lime.lime_text import LimeTextExplainer
from rouge_score import rouge_scorer
import torch, transformers

# -----------------------------
# Suppress warnings & optimize CPU
# -----------------------------
transformers.logging.set_verbosity_error()
torch.set_num_threads(torch.get_num_threads())

# -----------------------------
# Load models
# -----------------------------
summarizer_distilbart = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
summarizer_bart_large = None  # Lazy load BART-large
summarizer_t5 = pipeline("summarization", model="t5-small", device=-1)

qa_distilbert = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad", device=-1)
qa_roberta = pipeline("question-answering", model="deepset/roberta-base-squad2", device=-1)

kw_model = KeyBERT('all-MiniLM-L6-v2')
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Cache for BM25 embeddings per document
embeddings_cache = {}

sample_eval = [{"question":"What is AI?", "answer":"Artificial Intelligence is the simulation of human intelligence in machines."}]

# -----------------------------
# Utility functions
# -----------------------------
def extract_text(file_path):
    if str(file_path).lower().endswith(".pdf"):
        doc = fitz.open(str(file_path))
        return "\n".join([page.get_text("text") for page in doc])
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

def chunk_text(text, max_tokens=700):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, current_chunk, token_count = [], [], 0
    for s in sentences:
        tokens = s.split()
        if token_count + len(tokens) > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk, token_count = [], 0
        current_chunk.extend(tokens)
        token_count += len(tokens)
    if current_chunk: chunks.append(" ".join(current_chunk))
    return [c for c in chunks if len(c.split())>20]

def extract_keywords(text, top_n=15):
    return [kw[0] for kw in kw_model.extract_keywords(text, keyphrase_ngram_range=(1,2), stop_words='english', top_n=top_n)]

def generate_wordcloud(text):
    wc = WordCloud(width=600, height=300, background_color='white').generate(text)
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    wc.to_file(tmp_file.name)
    return tmp_file.name

def retrieve_passages(query, chunks, top_k=1, doc_id=None):
    # Precompute BM25 for the document if not cached
    if doc_id not in embeddings_cache:
        tokenized_chunks = [c.split() for c in chunks]
        bm25 = BM25Okapi(tokenized_chunks)
        embeddings_cache[doc_id] = bm25
    else:
        bm25 = embeddings_cache[doc_id]
    scores = bm25.get_scores(query.split())
    top_indices = scores.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

# -----------------------------
# Summarization
# -----------------------------
def summarize_document(files, model_choice, quality_choice):
    text = "".join([extract_text(f) + "\n" for f in files])
    chunks = chunk_text(text, max_tokens=700)
    global summarizer_bart_large

    # Determine summarizer & two-pass
    if model_choice=="T5":
        summarizer = summarizer_t5
        two_pass = False
    elif quality_choice=="High-Quality BART-large":
        if summarizer_bart_large is None:
            summarizer_bart_large = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
        summarizer = summarizer_bart_large
        two_pass = True
    else:
        summarizer = summarizer_distilbart
        two_pass = len(chunks) > 12

    # First pass
    summaries = [summarizer(c, max_length=200, min_length=100, do_sample=False, truncation=True)[0]['summary_text'] for c in chunks]

    # Second pass if needed
    if two_pass and len(summaries) > 1:
        final_summary = summarizer(" ".join(summaries), max_length=250, min_length=150, do_sample=False, truncation=True)[0]['summary_text']
    else:
        final_summary = " ".join(summaries)

    # Top sentences explainability
    doc_sentences = re.split(r'(?<=[.!?]) +', text)
    summary_emb = embedder.encode([final_summary])
    sentence_embs = embedder.encode(doc_sentences)
    sim_scores = np.dot(sentence_embs, summary_emb.T).flatten()
    top_sentences = [doc_sentences[i] for i in sim_scores.argsort()[-5:][::-1]]

    # Keywords & Wordcloud
    keywords = extract_keywords(final_summary)
    wc_file = generate_wordcloud(text)

    # ROUGE metrics
    scorer = rouge_scorer.RougeScorer(['rouge1','rougeL'], use_stemmer=True)
    score_list = [scorer.score(c, final_summary) for c in chunks]
    avg_rouge1 = np.mean([s['rouge1'].fmeasure for s in score_list])
    avg_rougeL = np.mean([s['rougeL'].fmeasure for s in score_list])
    rouge_text = f"Avg ROUGE-1: {avg_rouge1:.2f}, Avg ROUGE-L: {avg_rougeL:.2f}"

    # Save summary & combined report
    tmp_summary = tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8', suffix=".txt")
    tmp_summary.write(final_summary)
    tmp_summary.close()

    tmp_report = tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8', suffix=".txt")
    tmp_report.write("=== Summary ===\n"+final_summary+"\n\n")
    tmp_report.write("=== ROUGE Metrics ===\n"+rouge_text+"\n\n")
    tmp_report.write("=== Top Influential Sentences ===\n"+"\n".join(top_sentences)+"\n\n")
    tmp_report.write("=== Keywords ===\n"+", ".join(keywords)+"\n")
    tmp_report.write("=== Wordcloud ===\nSaved at: "+wc_file+"\n")
    tmp_report.close()

    return final_summary, ", ".join(keywords), wc_file, tmp_summary.name, rouge_text, "\n".join(top_sentences), tmp_report.name

# -----------------------------
# Question Answering
# -----------------------------
def ask_question(files, question, qa_model_choice, coverage_mode):
    text = "".join([extract_text(f) + "\n" for f in files])
    chunks = chunk_text(text, max_tokens=700)
    doc_id = str(hash(text))  # unique ID for caching embeddings

    # top_k based on coverage_mode
    top_k = 1 if coverage_mode=="Fast Mode" else 5
    top_passages = retrieve_passages(question, chunks, top_k=top_k, doc_id=doc_id)

    qa_model = qa_distilbert if qa_model_choice=="DistilBERT" else qa_roberta
    answers = [(qa_model(question=question, context=p)['answer'], qa_model(question=question, context=p)['score'], p) for p in top_passages]

    # Merge answers for full coverage mode
    if coverage_mode=="Full Coverage":
        merged_answer = " | ".join(list(set([a[0] for a in answers])))
        best_answer = (merged_answer, max(answers, key=lambda x:x[1])[1], "")
    else:
        best_answer = max(answers, key=lambda x:x[1])

    # Optional LIME explainability for small docs
    lime_exp = None
    if len(chunks)<6:
        explainer = LimeTextExplainer(class_names=["answer"])
        def predict_fn(texts): return [[qa_model(question=question, context=t)['score']] for t in texts]
        lime_exp = explainer.explain_instance(best_answer[2], predict_fn, num_features=10)

    keywords = extract_keywords(best_answer[2]) if best_answer[2] else []

    # EM/F1 evaluation
    em=f1=None
    for sample in sample_eval:
        if sample['question'].lower()==question.lower():
            pred = best_answer[0].lower()
            true = sample['answer'].lower()
            em = int(pred==true)
            pred_tokens, true_tokens = pred.split(), true.split()
            common = set(pred_tokens) & set(true_tokens)
            f1 = 2*len(common)/(len(pred_tokens)+len(true_tokens)) if len(common)>0 else 0
    eval_text = f"Exact Match: {em}, F1: {f1}" if em is not None else ""

    tmp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8', suffix=".txt")
    tmp_file.write(best_answer[0])
    tmp_file.close()

    return best_answer[0], ", ".join(keywords), lime_exp, tmp_file.name, eval_text

# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks() as demo:
    gr.Markdown("## Smart Document Summarizer & Q&A Assistant")

    with gr.Tab("Summarization"):
        files = gr.File(file_types=[".pdf",".txt"], file_count="multiple")
        model_choice = gr.Radio(["BART","T5"], label="Summarization Model")
        quality_choice = gr.Radio(["Fast CPU DistilBART","High-Quality BART-large"], label="Quality Mode")

        def toggle_quality(model):
            if model=="T5":
                return gr.update(choices=["Fast CPU T5-small"], value="Fast CPU T5-small", interactive=False)
            else:
                return gr.update(choices=["Fast CPU DistilBART","High-Quality BART-large"], value="Fast CPU DistilBART", interactive=True)
        model_choice.change(toggle_quality, inputs=[model_choice], outputs=[quality_choice])

        summary_out = gr.Textbox(label="Summary", lines=15)
        keywords_out = gr.Textbox(label="Top Keywords")
        wc_out = gr.Image(label="Wordcloud")
        download_summary = gr.File(label="Download Summary")
        rouge_textbox = gr.Textbox(label="ROUGE Scores")
        top_sentences_out = gr.Textbox(label="Top Influential Sentences")
        combined_report = gr.File(label="Download Combined Report")
        summarize_btn = gr.Button("Generate Summary")
        summarize_btn.click(
            summarize_document,
            inputs=[files, model_choice, quality_choice],
            outputs=[summary_out, keywords_out, wc_out, download_summary, rouge_textbox, top_sentences_out, combined_report]
        )

    with gr.Tab("Question Answering"):
        qa_files = gr.File(file_types=[".pdf",".txt"], file_count="multiple")
        question_input = gr.Textbox(label="Enter your question")
        qa_model = gr.Radio(["DistilBERT","RoBERTa"], label="QA Model")
        coverage_mode = gr.Radio(["Fast Mode","Full Coverage"], label="QA Coverage Mode")
        answer_out = gr.Textbox(label="Answer")
        qa_keywords_out = gr.Textbox(label="Context Keywords")
        lime_out = gr.JSON(label="LIME Explanation")
        download_qa = gr.File(label="Download QA Answer")
        eval_textbox = gr.Textbox(label="EM/F1 Evaluation")
        ask_btn = gr.Button("Ask Question")
        ask_btn.click(
            ask_question,
            inputs=[qa_files, question_input, qa_model, coverage_mode],
            outputs=[answer_out, qa_keywords_out, lime_out, download_qa, eval_textbox]
        )

demo.launch()
