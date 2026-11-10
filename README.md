# IntelliAssist AI: Smart Document AI Assistant

Ask questions about your PDF, TXT and DOCX files. Every answer shows the exact passages it came from, a confidence score, and refuses to guess when the answer isn't in your documents.

## What makes it different
- **Hybrid retrieval**: semantic search (BERT-family MiniLM embeddings + FAISS) fused with TF-IDF keyword search using Reciprocal Rank Fusion.
- **Proof, not promises**: cited passages with page numbers and highlighted matching terms.
- **Grounding guard**: if similarity is too low, the app says "not found" instead of hallucinating.
- **Follow-up questions** understood through conversation history.
- **Works without an API key** (extractive mode); add a Gemini key for generative answers.
- **Measured**: `evaluate.py` compares TF-IDF vs Dense vs Hybrid (Recall@k, MRR, latency).

## Architecture
Upload -> text extraction (pypdf / python-docx) -> LangChain RecursiveCharacterTextSplitter -> MiniLM embeddings -> FAISS
Question -> (history-aware rewrite) -> dense + TF-IDF ranking -> RRF fusion -> top-k passages -> Gemini (or extractive) -> answer + citations + confidence

## Run locally
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
export GEMINI_API_KEY=your_key                        # optional; Windows: set GEMINI_API_KEY=your_key
streamlit run app.py
```

## Evaluate
Put 3-5 documents in `sample_docs/`, write 15-20 questions in `eval_set.json`, then `python evaluate.py`.

## Deploy (Streamlit Community Cloud)
Push to GitHub -> share.streamlit.io -> New app -> `app.py` -> Settings > Secrets: `GEMINI_API_KEY = "..."`

## Tech stack
Streamlit, LangChain, Hugging Face (sentence-transformers, DistilBERT sentiment), FAISS, scikit-learn TF-IDF, Gemini API.

## Extra features
- **Quiz generator**: multiple-choice quiz from any uploaded document (Gemini, or offline fill-in-the-blank fallback).
- **Cross-encoder re-ranking** (toggle) for sharper top-1 accuracy on English documents.
- **Multilingual mode**: ask in Hindi, Kannada, Tamil etc. over English documents.
- `sample_docs/rag_basics.txt` + `eval_set.json` are a smoke test only. Replace with your own 3-5 documents and 15-20 questions for the real evaluation.
