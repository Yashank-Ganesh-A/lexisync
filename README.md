# LexiSyncAI — Student Implementation B

Python + Streamlit legal-document analyst following the supplied DA3 brief.

## Run on Windows
```powershell
py -3.11 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env and set GROQ_API_KEY
streamlit run app.py
```
Groq key: https://console.groq.com/keys. Default model: `openai/gpt-oss-120b`. Check current availability at https://console.groq.com/docs/models. Groq is API inference, not local LLM inference.

## Demo
Upload 2–5 public PDFs → extract → inspect structure → build hybrid index → ask clause/factual question → inspect cited passages → ask an unanswerable question → run benchmark.

## Brief coverage
- PDF ingestion and reading-order extraction: PyMuPDF sorted blocks.
- Scanned-page detection and optional Tesseract OCR fallback.
- Clause/section-aware segmentation with source page metadata.
- BM25 exact lexical retrieval plus SentenceTransformer dense embeddings.
- Reciprocal Rank Fusion and cross-encoder reranking.
- Groq generation from retrieved context only.
- Page/section evidence panel, confidence label and abstention gate.
- 16-question benchmark scaffold including four unanswerable prompts.
- Dense retrieval and hybrid retrieval modes exposed for comparison.
- Latency reporting and downloadable benchmark JSON.

## Required work before claiming final metrics
The starter questions are templates, not a validated held-out benchmark. Build document-specific questions, expected answers and gold passage IDs. Include factual, multi-page multi-hop and at least three unanswerable questions. Manually grade correctness and citation precision. Calibrate abstention on validation data. Run dense-only vs hybrid on the same test set and report actual Recall@k and latency.

## Limitations
OCR requires the Tesseract executable. Complex columns/footnotes need manual validation. Table detection is a count only; merged-cell extraction is not implemented. Citations are page/section level; precise bounding-box citation is not implemented. Abstention uses an overlap heuristic and must be calibrated. Cross-document conflict output needs manual verification. Groq is cloud API inference, not local inference.
\n## Interface edition B\nThe second implementation uses a warm paper-and-charcoal casework-console design, numbered workflow tabs, a compact editorial masthead, and a dark control panel. It is intentionally distinct from the first copy's blue/teal workspace presentation.\n