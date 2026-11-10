"""IntelliAssist AI - core RAG engine (loading, chunking, hybrid retrieval, answering, NLP)."""
import io, json, os, random, re
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"  # negative / neutral / positive
MIN_DENSE = 0.22  # below this cosine similarity we refuse to answer (anti-hallucination guard)


MULTI_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_ce = None


def rerank_hits(query, hits):
    """Cross-encoder re-ranking: reads question + passage together for sharper relevance."""
    global _ce
    if _ce is None:
        from sentence_transformers import CrossEncoder
        _ce = CrossEncoder(RERANK_MODEL)
    sc = _ce.predict([(query, h["doc"].page_content) for h in hits])
    for h, s in zip(hits, sc):
        h["rerank"] = float(s)
    return sorted(hits, key=lambda h: h["rerank"], reverse=True)


def get_embedder(name=EMB_MODEL):
    return HuggingFaceEmbeddings(model_name=name, encode_kwargs={"normalize_embeddings": True})


# ---------- 1. Loading ----------
def _clean(t):
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", t)).strip()


def load_file(name: str, data: bytes) -> list[Document]:
    ext = name.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        from pypdf import PdfReader
        pages = [(i + 1, p.extract_text() or "") for i, p in enumerate(PdfReader(io.BytesIO(data)).pages)]
    elif ext == "docx":
        import docx
        pages = [(1, "\n".join(p.text for p in docx.Document(io.BytesIO(data)).paragraphs))]
    else:
        pages = [(1, data.decode("utf-8", "ignore"))]
    return [Document(page_content=_clean(t), metadata={"source": name, "page": n}) for n, t in pages if t.strip()]


# ---------- 2. Knowledge base with hybrid retrieval ----------
class KnowledgeBase:
    def __init__(self, embedder, chunk_size=700, overlap=120, multilingual=False):
        self.embedder, self.chunk_size, self.overlap = embedder, chunk_size, overlap
        self.multilingual = multilingual
        self.pages, self.chunks, self.vs = [], [], None

    def build(self, pages: list[Document]):
        self.pages = pages
        splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.overlap)
        self.chunks = splitter.split_documents(pages)
        for i, c in enumerate(self.chunks):
            c.metadata["cid"] = i
        self.vs = FAISS.from_documents(self.chunks, self.embedder)
        self.tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.mat = self.tfidf.fit_transform([c.page_content for c in self.chunks])
        return self

    @property
    def sources(self):
        return sorted({c.metadata["source"] for c in self.chunks})

    def search(self, query, k=4, mode="hybrid", sources=None, rerank=False):
        n = len(self.chunks)
        dense = {d.metadata["cid"]: 1 - float(s) / 2  # L2^2 on unit vectors -> cosine
                 for d, s in self.vs.similarity_search_with_score(query, k=min(n, 100))}
        tf = linear_kernel(self.tfidf.transform([query]), self.mat)[0]
        r_dense = sorted(dense, key=dense.get, reverse=True)
        r_tf = [int(i) for i in np.argsort(tf)[::-1]]
        if mode == "dense":
            order = r_dense
        elif mode == "tfidf":
            order = r_tf
        else:  # Reciprocal Rank Fusion of both rankings
            sc = defaultdict(float)
            for ranking in (r_dense, r_tf):
                for rank, cid in enumerate(ranking):
                    sc[cid] += 1 / (60 + rank)
            order = sorted(sc, key=sc.get, reverse=True)
        use_rr = rerank and not self.multilingual  # cross-encoder is English-only
        limit = k * 3 if use_rr else k
        hits = []
        for cid in order:
            c = self.chunks[cid]
            if sources and c.metadata["source"] not in sources:
                continue
            hits.append({"doc": c, "dense": dense.get(cid, 0.0), "tfidf": float(tf[cid])})
            if len(hits) == limit:
                break
        if use_rr and len(hits) > 1:
            hits = rerank_hits(query, hits)
        return hits[:k]


# ---------- 3. NLP helpers ----------
_INTENTS = [
    ("summarize", r"\b(summar\w*|overview|tl;?dr|gist|main points?)\b"),
    ("compare", r"\b(compare|comparison|difference|differ\w*|versus|vs\.?)\b"),
    ("list", r"\b(list|enumerate|what are|steps|types of|examples?)\b"),
    ("define", r"\b(what is|what's|define|definition|meaning of)\b"),
    ("explain", r"\b(how|why|explain)\b"),
]


def detect_intent(q: str) -> str:
    for label, pat in _INTENTS:
        if re.search(pat, q.lower()):
            return label
    return "fact lookup"


_sent = None


def sentiment(texts):
    """Returns (label, mean_positive_probability) over a list of texts (3-class: Positive/Neutral/Negative)."""
    global _sent
    if _sent is None:
        from transformers import pipeline
        _sent = pipeline("sentiment-analysis", model=SENT_MODEL, top_k=None)
    outs = _sent([t[:1500] for t in texts], truncation=True, max_length=512)
    if outs and isinstance(outs[0], dict):  # older/newer pipelines may return a flat list
        outs = [outs]
    tot = {}
    for o in outs:
        for d in o:
            tot[d["label"].lower()] = tot.get(d["label"].lower(), 0) + d["score"] / len(outs)
    label = max(tot, key=tot.get)
    return label.capitalize(), float(tot.get("positive", 0.0))


def keywords(texts, n=10):
    v = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=2000, sublinear_tf=True)
    m = v.fit_transform(texts)
    sc = np.asarray(m.sum(axis=0)).ravel()
    terms = v.get_feature_names_out()
    return [(terms[i], float(sc[i])) for i in np.argsort(sc)[::-1][:n]]


def _sentences(t):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", t) if len(s.strip()) > 25]


def extractive(text, query=None, n=3):
    """No-API fallback: pick the n most relevant/central sentences."""
    s = _sentences(text)
    if len(s) <= n:
        return " ".join(s) or text[:500]
    try:
        v = TfidfVectorizer(stop_words="english").fit(s + ([query] if query else []))
        m = v.transform(s)
        sc = linear_kernel(v.transform([query]), m)[0] if query else np.asarray(m.sum(axis=1)).ravel()
    except ValueError:
        return " ".join(s[:n])
    return " ".join(s[i] for i in sorted(np.argsort(sc)[::-1][:n]))


# ---------- 4. Generation (Gemini, with offline fallback) ----------
_clients = {}


def llm(prompt, api_key):
    from google import genai
    client = _clients.get(api_key)  # keep a reference, otherwise the client is closed mid-call
    if client is None:
        client = _clients[api_key] = genai.Client(api_key=api_key)
    import time
    last = None
    # try the main model twice, then a lighter backup twice (handles 503 "high demand" spikes)
    for model in [os.getenv("GEMINI_MODEL", "gemini-3.8-flash")] * 2 + ["gemini-3.5-flash-lite"] * 2:
        try:
            return (client.models.generate_content(model=model, contents=prompt).text or "").strip()
        except Exception as e:
            last = e
            time.sleep(2)
    raise last


def condense(question, history, api_key):
    """Rewrite a follow-up into a standalone question using chat history."""
    if not (api_key and history):
        return question
    h = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in history[-4:])
    try:
        return llm(f"Chat history:\n{h}\n\nRewrite the final question as a standalone search query. "
                   f"Output only the query.\nQuestion: {question}", api_key) or question
    except Exception:
        return question


def answer(question, hits, history, api_key, min_dense=MIN_DENSE, user_request=None):
    """Returns (answer_text, confidence 0-1, mode_used)."""
    top = max((h["dense"] for h in hits), default=0.0)
    conf = float(np.clip((top - 0.15) / 0.45, 0, 1))
    if not hits or top < min_dense:
        return ("I couldn't find this in your documents, so I won't guess. "
                "Try rephrasing, or upload the document that covers it."), 0.0, "guard"
    ctx = "\n\n".join(f"[{i+1}] ({h['doc'].metadata['source']}, p.{h['doc'].metadata['page']})\n"
                      f"{h['doc'].page_content}" for i, h in enumerate(hits))
    if api_key:
        hist = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in history[-4:])
        prompt = ("You are IntelliAssist, a document assistant. Answer ONLY from the context. "
                  "Write a clear, complete answer in 2-5 sentences (short bullets for lists) and cite sources inline like [1], [2]. "
                  "If the user asks for a simpler, shorter or different style, follow that style while staying faithful to the context. "
                  "If the context is insufficient, say so. Reply in the same language as the question.\n\n"
                  f"Conversation:\n{hist}\n\nContext:\n{ctx}\n\n"
                  f"User message: {user_request or question}\nStandalone question: {question}\nAnswer:")
        try:
            return llm(prompt, api_key), conf, "gemini"
        except Exception as e:
            return extractive(ctx, question) + f"\n\n(LLM unavailable: {type(e).__name__}: {str(e)[:300]}; showing extractive answer.)", conf, "extractive"
    return extractive(ctx, question, n=4), conf, "extractive"


def summarize(text, length, api_key):
    words = {"Short": "about 3 sentences", "Medium": "one paragraph plus 5 bullet points",
             "Detailed": "a structured summary with headings and bullets"}[length]
    if api_key:
        try:
            return llm(f"Summarize the document below in {words}. Be faithful, no outside facts.\n\n{text[:30000]}", api_key)
        except Exception:
            pass
    return extractive(text, None, n={"Short": 3, "Medium": 6, "Detailed": 10}[length])


# ---------- 5. Quiz generator ----------
def make_quiz(text, n, api_key):
    """Multiple-choice quiz. Gemini if a key is set, otherwise fill-in-the-blank from the text."""
    if api_key:
        try:
            raw = llm(f"Create {n} multiple-choice questions from the text. Return ONLY a JSON list like "
                      '[{"question":"...","options":["A","B","C","D"],"answer":0,"explanation":"..."}] '
                      f"where answer is the 0-based index of the correct option. Use only facts in the text.\n\n{text[:20000]}",
                      api_key)
            qs = json.loads(re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip())
            good = [q for q in qs if len(q.get("options", [])) == 4 and q.get("answer") in range(4)]
            if good:
                return good[:n]
        except Exception:
            pass
    return _cloze_quiz(text, n)


def _cloze_quiz(text, n):
    sents = [s for s in _sentences(text) if 40 < len(s) < 220]
    terms = [t for t, _ in keywords([text], 80) if " " not in t and len(t) > 4]
    rnd, out = random.Random(7), []
    for s in sents:
        hit = next((t for t in terms if re.search(rf"\b{re.escape(t)}\b", s, re.I)), None)
        pool = [t for t in terms if t != hit]
        if not hit or len(pool) < 3:
            continue
        opts = rnd.sample(pool, 3) + [hit]
        rnd.shuffle(opts)
        out.append({"question": "Fill in the blank: " + re.sub(rf"\b{re.escape(hit)}\b", "_____", s, count=1, flags=re.I),
                    "options": opts, "answer": opts.index(hit), "explanation": "From the document: " + s})
        if len(out) == n:
            break
    return out
