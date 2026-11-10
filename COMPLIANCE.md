# Compliance with the Major Project brief (Launched Global - AI)

| Brief requirement | Where it is met |
|---|---|
| Document Upload (PDF/TXT/DOCX) | `core.load_file`, sidebar uploader (multiple files) |
| AI Chatbot with RAG | `app.py` Chat tab, `core.answer` (LangChain splitter + FAISS vector store) |
| Semantic Search using Embeddings | `KnowledgeBase.search` dense mode (MiniLM, a distilled BERT-family model) |
| Text Summarisation | Summarize tab, `core.summarize` (Short / Medium / Detailed) |
| Sentiment & Intent Analysis | `core.sentiment` (DistilBERT), `core.detect_intent`, Insights tab |
| Source Citation Display | Sources expander: file, page, highlighted passage, match score |
| Conversation History | Session chat history, history-aware follow-ups, JSON export |
| Streamlit Web Interface | `app.py` |
| Tech: Python, LangChain, Hugging Face, Gemini | `core.py` |
| NLP: TF-IDF, BERT embeddings | `TfidfVectorizer` + `sentence-transformers` |
| Vector DB: FAISS | `langchain_community.vectorstores.FAISS` |
| Deployment: Streamlit Cloud | README deploy section |
| Steps 1-7 | collect docs -> `load_file` -> embeddings/FAISS -> RAG -> chat+summary -> UI -> deploy+test (`evaluate.py`) |

## Deliverables checklist
- [ ] GitHub repository (this folder + screenshots in README)
- [ ] PPT presentation
- [ ] 3-5 minute demo video (email: show real screens, short, structured)
- [ ] Evaluation report (from `evaluation_results.csv`)
- [ ] Final project report
- [ ] Submit by 10 Nov 2026 via the "Submit Your Project Demonstration" button (check the guidelines PDF for exact instructions)
