# 📑 Smart Document Summarizer & Q&A Assistant

## Description
This project is a **full-featured NLP application** that allows users to **upload documents (PDF or TXT)** and perform:

1. **Document Summarization** – Fast or high-quality summaries using DistilBART, BART-large, or T5.  
2. **Question Answering (Q&A)** – Retrieve answers from the document using DistilBERT or RoBERTa with **Fast** or **Full Coverage** modes.  
3. **Explainability & Analytics** – Top influential sentences, keywords, wordcloud, ROUGE evaluation, and optional LIME explanations.  
4. **Downloadable outputs** – Summary, combined report, and QA answers for portfolio/demo purposes.

This project is **placement-ready**, **resume-worthy**, and optimized to run **entirely on CPU**.

---

## Features

### Summarization
- Select summarization model: **BART**, **T5**  
- Quality modes:
  - **Fast CPU DistilBART** – optimized for speed  
  - **High-Quality BART-large** – slower but more accurate  
- Optional **two-pass summarization** for long documents  
- Outputs:
  - Summary text  
  - Top influential sentences  
  - Keywords  
  - Wordcloud image  
  - ROUGE evaluation metrics  
  - Combined report download  

### Question Answering (Q&A)
- Select QA model: **DistilBERT** (fast) or **RoBERTa** (higher quality)  
- Coverage mode:
  - **Fast Mode** → top_k=1, <2 min for 11-page PDFs  
  - **Full Coverage** → top_k=5, retrieves all mentions/models (slightly slower)  
- Optional **LIME explanations** for small PDFs  
- Outputs:
  - Answer text  
  - Context keywords  
  - LIME JSON explanation  
  - Download QA answer  
  - EM/F1 evaluation for sample questions  

### Explainability & Analytics
- Top sentences from the document influencing the summary  
- Extracted keywords from summary and QA context  
- Wordcloud visualization of document content  
- ROUGE evaluation for summary quality  

---

## Author
**Harthik Manichandra Vanumu** 

**Made with ❤️ by Harthik**
