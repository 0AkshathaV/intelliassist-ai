"""Retrieval evaluation: TF-IDF vs Dense vs Hybrid.  Run: python evaluate.py
Put your documents in sample_docs/ and fill eval_set.json with questions + keywords that
must appear in the correct passage.  Results feed your Evaluation Report directly."""
import json, pathlib, time
import pandas as pd
from core import KnowledgeBase, get_embedder, load_file

docs = []
for p in sorted(pathlib.Path("sample_docs").glob("*")):
    if p.suffix.lower() in {".pdf", ".txt", ".docx"}:
        docs += load_file(p.name, p.read_bytes())
kb = KnowledgeBase(get_embedder()).build(docs)
qs = json.load(open("eval_set.json"))
K = 5
rows = []
for mode, rr in [("tfidf", False), ("dense", False), ("hybrid", False), ("hybrid", True)]:
    ranks, t0 = [], time.time()
    for item in qs:
        hits = kb.search(item["question"], K, mode, rerank=rr)
        r = next((i + 1 for i, h in enumerate(hits)
                  if any(" ".join(k.lower().split()) in " ".join(h["doc"].page_content.lower().split()) for k in item["keywords"])), None)
        ranks.append(r)
    n = len(qs)
    rows.append({"Method": mode + ("+rerank" if rr else ""),
                 "Recall@1": sum(r == 1 for r in ranks) / n,
                 "Recall@3": sum(r is not None and r <= 3 for r in ranks) / n,
                 "Recall@5": sum(r is not None for r in ranks) / n,
                 "MRR": sum(1 / r for r in ranks if r) / n,
                 "Avg latency (ms)": round((time.time() - t0) / n * 1000, 1)})
df = pd.DataFrame(rows).round(3)
print(df.to_markdown(index=False))
df.to_csv("evaluation_results.csv", index=False)
