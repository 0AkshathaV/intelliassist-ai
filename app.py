"""IntelliAssist AI - Streamlit UI. Run: streamlit run app.py"""
import html, json, os, re
import pandas as pd
import streamlit as st
from core import (KnowledgeBase, get_embedder, load_file, detect_intent, sentiment, keywords,
                  condense, answer, summarize, make_quiz, EMB_MODEL, MULTI_MODEL)

st.set_page_config(page_title="IntelliAssist AI", page_icon="📑", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
:root { --ink:#101828; --muted:#667085; --line:#E4E7EC; --brand:#1E4E8C; --brand-dark:#173F72; --bg:#F4F6F9; }
html { font-size:17px; }
html, body, [class*="css"] { font-family:'IBM Plex Sans',sans-serif; color:var(--ink); }
h1, h2, h3 { font-family:'Source Serif 4',serif !important; color:#0B2545; letter-spacing:-.01em; }
.stApp { background:var(--bg); }
.block-container { padding-top:2rem; max-width:1100px; }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stBottom"] > div { background:transparent; }
.stAppDeployButton, #MainMenu, footer { display:none !important; }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] { display:none !important; }
[data-testid="stElementContainer"]:has(.mu), [data-testid="stElementContainer"]:has(.ma) { display:none; }
.hero { background:linear-gradient(120deg,#0B2545 0%,#13315C 100%); color:#fff; border-radius:14px; padding:2rem 2.2rem;
        margin-bottom:1.4rem; box-shadow:0 8px 24px rgba(11,37,69,.18); }
.eyebrow { letter-spacing:.14em; font-size:.75rem; text-transform:uppercase; color:#9DB7DD; font-weight:600; margin-bottom:.5rem; }
.hero .t { font-family:'Source Serif 4',serif; font-size:2.5rem; font-weight:700; line-height:1.15; letter-spacing:-.02em; color:#fff; }
.hero .s { color:#C7D6EC; margin-top:.45rem; font-size:1.05rem; }
.pill { display:inline-block; background:rgba(255,255,255,.10); border:1px solid rgba(255,255,255,.22); color:#E6EEF9;
        border-radius:6px; padding:3px 12px; font-size:.85rem; margin:14px 8px 0 0; }
.step { background:#fff; border:1px solid var(--line); border-radius:12px; padding:1.2rem 1.3rem; min-height:140px;
        box-shadow:0 1px 3px rgba(16,24,40,.06); }
.step .n { display:inline-flex; width:30px; height:30px; border-radius:8px; background:#EAF1FB; color:var(--brand);
           align-items:center; justify-content:center; font-weight:700; margin-bottom:.6rem; }
.step .h { font-size:1.2rem; font-weight:600; color:#0B2545; }
.step .d { color:var(--muted); font-size:.98rem; margin-top:.25rem; }
[data-baseweb="tab-list"] { gap:8px; }
button[data-baseweb="tab"] { font-weight:600; font-size:1rem; color:var(--muted); background:transparent; }
button[data-baseweb="tab"][aria-selected="true"] { color:var(--brand); }
[data-baseweb="tab-highlight"] { background:var(--brand) !important; height:3px !important; }
[data-baseweb="tab-border"] { background:var(--line) !important; }
[data-testid="stChatMessage"] { background:#fff; border:1px solid var(--line); border-radius:12px; padding:1rem 1.1rem;
        margin-bottom:.7rem; box-shadow:0 1px 3px rgba(16,24,40,.05); }
[data-testid="stChatMessage"]:has(.mu) { background:#EEF3FA; border-color:#D5E1F3; }
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li { font-size:1.05rem; line-height:1.65; }
[data-testid="stChatInput"] { border-radius:12px; background:#fff; border:1px solid #D0D5DD; box-shadow:0 4px 14px rgba(16,24,40,.08); }
.stButton>button, .stDownloadButton>button { border-radius:8px; font-weight:600; transition:all .15s ease; }
.stButton>button[kind="primary"] { background:var(--brand); border:1px solid var(--brand); color:#fff; }
.stButton>button[kind="primary"]:hover { background:var(--brand-dark); border-color:var(--brand-dark); }
.stButton>button[kind="secondary"], .stDownloadButton>button { background:#fff; border:1px solid #D0D5DD; color:#344054; }
.stButton>button[kind="secondary"]:hover, .stDownloadButton>button:hover { background:#F9FAFB; border-color:#98A2B3; }
input, textarea, [data-baseweb="select"] > div { border-radius:8px !important; }
[data-testid="stExpander"] details, [data-testid="stVerticalBlockBorderWrapper"] { background:#fff; border:1px solid var(--line) !important; border-radius:12px; }
[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:12px; padding:.9rem 1.1rem; box-shadow:0 1px 3px rgba(16,24,40,.05); }
[data-testid="stMetricValue"] { color:#0B2545; }
.src { background:#fff; border:1px solid var(--line); border-left:3px solid var(--brand); padding:.8rem 1rem;
       border-radius:4px 10px 10px 4px; margin:.5rem 0; font-size:.97rem; line-height:1.55; }
.src b { color:var(--brand); }
.bar { height:4px; background:#EAECF0; border-radius:3px; margin:7px 0 6px; }
.bar span { display:block; height:100%; background:var(--brand); border-radius:3px; }
mark { background:#FEF0C7; padding:0 2px; border-radius:2px; }
.chip { display:inline-block; padding:2px 10px; border-radius:6px; font-size:.82rem; margin-right:6px; background:#F2F4F7;
        color:#344054; border:1px solid var(--line); font-weight:500; }
.chip.good, .chip.tone-positive { background:#ECFDF3; color:#067647; border-color:#ABEFC6; }
.chip.warn { background:#FFFAEB; color:#B54708; border-color:#FEDF89; }
.chip.blue { background:#EFF8FF; color:#175CD3; border-color:#B2DDFF; }
.chip.violet { background:#F4F3FF; color:#5925DC; border-color:#D9D6FE; }
.chip.tone-neutral { background:#F2F4F7; color:#475467; }
.chip.tone-negative { background:#FEF3F2; color:#B42318; border-color:#FECDCA; }
.foot { text-align:center; color:#98A2B3; font-size:.85rem; margin-top:2.5rem; }
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading embedding model (first run only)…")
def embedder(name):
    return get_embedder(name)


def secret_key():
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return ""


ss = st.session_state
ss.setdefault("kb", None)
ss.setdefault("messages", [])

# ---------------- Hero ----------------
pills = "".join(f'<span class="pill">{html.escape(s)}</span>' for s in (ss.kb.sources if ss.kb else
                ["Hybrid search", "Cited answers", "Not-found guard"]))
st.markdown(f'<div class="hero"><div class="eyebrow">AI Document Assistant</div><div class="t">Ask your documents. See the proof.</div>'
            f'<div class="s">Answers come only from your files, with the exact passages behind every one.</div>'
            f'{pills}</div>', unsafe_allow_html=True)

# ---------------- Controls (replaces the sidebar) ----------------
box = st.container(border=True) if not ss.kb else st.expander("Documents and settings")
with box:
    files = st.file_uploader("Upload PDF, TXT or DOCX files", type=["pdf", "txt", "docx"],
                             accept_multiple_files=True, key="files")
    multi = st.checkbox("Multilingual mode (Hindi, Kannada, Tamil…)", key="multi",
                        help="Ask in any language. Uses a multilingual embedding model; re-index after changing.")
    if st.button("Index documents", type="primary", disabled=not files, key="index"):
        pages = []
        for f in files:
            pages += load_file(f.name, f.getvalue())
        if not pages:
            st.error("No readable text found. Scanned PDFs need OCR first.")
        else:
            with st.spinner("Reading, chunking and embedding…"):
                ss.kb = KnowledgeBase(embedder(MULTI_MODEL if multi else EMB_MODEL), multilingual=multi).build(pages)
            ss.messages = []
            ss["scope"] = []
            ss.flash = f"Indexed {len(ss.kb.sources)} file(s) into {len(ss.kb.chunks)} passages. Ask away!"
            st.rerun()
    st.markdown("**Settings**")
    c1, c2 = st.columns(2)
    mode = c1.selectbox("Search method", ["hybrid", "dense", "tfidf"], key="mode",
                        help="hybrid = semantic embeddings + TF-IDF keywords, fused with Reciprocal Rank Fusion")
    top_k = c2.slider("Passages to retrieve", 2, 8, 4, key="topk")
    c3, c4 = st.columns(2)
    api_key = c3.text_input("Gemini API key (optional)", type="password", key="apikey",
                            value=os.getenv("GEMINI_API_KEY", "") or secret_key(),
                            help="Without a key the app still works using extractive answers.")
    scope = c4.multiselect("Limit to files", ss.kb.sources if ss.kb else [], key="scope",
                           help="Empty = search all files")
    rerank = st.toggle("Re-rank with cross-encoder (English, slower)", value=False, key="rerank")
if "flash" in ss:
    st.success(ss.pop("flash"))

if not ss.kb:
    cols = st.columns(3)
    for c, (n, h, d) in zip(cols, [("1", "Upload", "Add PDF, TXT or DOCX files from the sidebar."),
                                    ("2", "Index", "Click Index documents to read and embed them."),
                                    ("3", "Ask", "Chat, summarise, take a quiz, and check every source.")]):
        c.markdown(f'<div class="step s{n}"><div class="n">{n}</div><div class="h">{h}</div><div class="d">{d}</div></div>',
                   unsafe_allow_html=True)
    st.markdown('<div class="foot">IntelliAssist AI · Akshatha V · LaunchED Global AI Major Project</div>',
                unsafe_allow_html=True)
    st.stop()

kb = ss.kb


def mark(text, query):
    t = html.escape(text)
    for w in {w for w in re.findall(r"[A-Za-z0-9]{4,}", query)}:
        t = re.sub(f"(?i)({re.escape(w)})", r"<mark>\1</mark>", t)
    return t


def render_sources(hits, query):
    for i, h in enumerate(hits, 1):
        m = h["doc"].metadata
        pct = max(0, min(100, round(h["dense"] * 100)))
        st.markdown(f'<div class="src c{i % 3}"><b>[{i}] {html.escape(m["source"])} · page {m["page"]}</b> · match {pct}%'
                    f'<div class="bar"><span style="width:{pct}%"></span></div>'
                    f'{mark(h["doc"].page_content, query)}</div>', unsafe_allow_html=True)


def render_msg(m):
    with st.chat_message(m["role"]):
        st.markdown(f'<span class="{"mu" if m["role"] == "user" else "ma"}"></span>', unsafe_allow_html=True)
        st.markdown(m["content"])
        if m["role"] == "assistant" and m.get("hits"):
            cls = "good" if m["conf"] > .6 else "warn" if m["conf"] < .3 else ""
            st.markdown(f'<span class="chip {cls}">confidence {m["conf"]:.0%}</span>'
                        f'<span class="chip blue">intent: {m["intent"]}</span>'
                        f'<span class="chip tone-{str(m["tone"]).lower()}">question tone: {m["tone"]}</span>'
                        f'<span class="chip violet">{m["engine"]}</span>', unsafe_allow_html=True)
            with st.expander(f"Sources ({len(m['hits'])})"):
                render_sources(m["hits"], m["query"])


tab_chat, tab_sum, tab_quiz, tab_ins = st.tabs(["Chat", "Summarize", "Quiz", "Insights"])

# ---------------- Chat ----------------
with tab_chat:
    if ss.messages:
        _, cd, cc = st.columns([4, 1.4, 1.2])
        cd.download_button("Download chat", json.dumps(ss.messages, indent=2, default=str),
                           "intelliassist_chat.json", width="stretch")
        if cc.button("Clear", width="stretch"):
            ss.messages = []
            st.rerun()
    for m in ss.messages:
        render_msg(m)
    if not ss.messages:
        st.caption("Try asking:")
        sg = st.columns(3)
        for i, (c, t) in enumerate(zip(sg, ["Summarise the main points of these documents",
                                            "What are the key terms and definitions?",
                                            "List the important numbers or rules mentioned"])):
            if c.button(t, key=f"sg{i}", width="stretch"):
                ss.pending = t
    q = st.chat_input("Ask a question about your documents")
    if not q:
        q = ss.pop("pending", None)
    if q:
        ss.messages.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown('<span class="mu"></span>', unsafe_allow_html=True)
            st.markdown(q)
        with st.chat_message("assistant"), st.spinner("Searching and reasoning…"):
            standalone = condense(q, ss.messages[:-1], api_key)
            hits = kb.search(standalone, top_k, mode, scope or None, rerank)
            text, conf, engine = answer(standalone, hits, ss.messages[:-1], api_key,
                                        min_dense=0.12 if kb.multilingual else 0.22, user_request=q)
            tone = sentiment([q])[0]
        ss.messages.append({"role": "assistant", "content": text, "hits": hits if engine != "guard" else [],
                            "conf": conf, "intent": detect_intent(q), "tone": tone,
                            "engine": engine, "query": standalone})
        st.rerun()

# ---------------- Summarize ----------------
with tab_sum:
    c1, c2 = st.columns([2, 1])
    target = c1.selectbox("Document", kb.sources)
    length = c2.radio("Length", ["Short", "Medium", "Detailed"], horizontal=True, index=1)
    if st.button("Summarize document", type="primary"):
        text = "\n".join(p.page_content for p in kb.pages if p.metadata["source"] == target)
        with st.spinner("Summarizing…"):
            with st.container(border=True):
                st.markdown(summarize(text, length, api_key))

# ---------------- Quiz ----------------
with tab_quiz:
    c1, c2 = st.columns([2, 1])
    qdoc = c1.selectbox("Document", kb.sources, key="quiz_doc")
    nq = c2.slider("Questions", 3, 10, 5)
    if st.button("Generate quiz", type="primary"):
        for k in [k for k in ss if k.startswith("qz")]:
            del ss[k]
        text = "\n".join(p.page_content for p in kb.pages if p.metadata["source"] == qdoc)
        with st.spinner("Writing questions…"):
            ss.quiz = make_quiz(text, nq, api_key)
    if "quiz" in ss and not ss.quiz:
        st.warning("Not enough text to build a quiz from this document.")
    for i, qz in enumerate(ss.get("quiz", [])):
        with st.container(border=True):
            st.markdown(f"**{i + 1}. {qz['question']}**")
            st.radio("Answer", qz["options"], index=None, key=f"qz{i}", label_visibility="collapsed")
    if ss.get("quiz") and st.button("Check answers"):
        score = 0
        for i, qz in enumerate(ss.quiz):
            right = qz["options"][qz["answer"]]
            ok = ss.get(f"qz{i}") == right
            score += ok
            (st.success if ok else st.error)(
                f"Q{i + 1}: {'Correct' if ok else 'Incorrect. Correct answer: ' + right}. {qz['explanation']}")
        st.metric("Score", f"{score}/{len(ss.quiz)}")

# ---------------- Insights ----------------
with tab_ins:
    rows = []
    for s in kb.sources:
        cs = [c.page_content for c in kb.chunks if c.metadata["source"] == s]
        label, pos = sentiment(cs[:25])
        rows.append({"Document": s, "Passages": len(cs), "Words": sum(len(c.split()) for c in cs),
                     "Overall tone": label, "Positivity": round(pos, 2)})
    m1, m2, m3 = st.columns(3)
    m1.metric("Documents", len(kb.sources))
    m2.metric("Passages indexed", len(kb.chunks))
    m3.metric("Total words", f"{sum(r['Words'] for r in rows):,}")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.subheader("Key topics")
    kw = sorted(keywords([c.page_content for c in kb.chunks]), key=lambda t: t[1])
    st.bar_chart(pd.DataFrame(kw, columns=["term", "weight"]).set_index("term"), horizontal=True, color="#0F6B6B")

st.markdown('<div class="foot">IntelliAssist AI · Akshatha V · LaunchED Global AI Major Project</div>',
            unsafe_allow_html=True)
