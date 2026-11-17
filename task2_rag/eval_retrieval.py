"""
Recall@k for each chunk size, using the 15 hand-written questions in questions.json.

A retrieval counts as a HIT if any of the top-k chunks contains the gold "answer_span"
(a short exact phrase copied from the source passage), compared after normalising whitespace/case.
This is stricter and more honest than matching on page number alone.

    python eval_retrieval.py --sizes 128 512 --k 5
Writes results/retrieval_eval.json (per-question detail, use it to find the 2 failure cases).
"""
import argparse, json, re
from pathlib import Path

from rag import search

HERE = Path(__file__).parent
norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()

p = argparse.ArgumentParser()
p.add_argument("--sizes", type=int, nargs="+", default=[128, 512])
p.add_argument("--k", type=int, default=5)
args = p.parse_args()

qs = json.load(open(HERE / "questions.json", encoding="utf-8"))
assert len(qs) >= 15, "write at least 15 questions"
report = {}

for size in args.sizes:
    hits_at = {1: 0, 3: 0, args.k: 0}
    detail = []
    for q in qs:
        res = search(q["question"], size, k=args.k)
        rank = next((i + 1 for i, (c, _) in enumerate(res) if norm(q["answer_span"]) in norm(c["text"])), None)
        for kk in hits_at:
            hits_at[kk] += rank is not None and rank <= kk
        detail.append({"question": q["question"], "gold_page": q.get("page"), "hit_rank": rank,
                       "retrieved_pages": [c["page"] for c, _ in res],
                       "top1_preview": res[0][0]["text"][:200]})
    n = len(qs)
    report[f"chunk{size}"] = {f"recall@{kk}": round(v / n, 3) for kk, v in hits_at.items()} | {"detail": detail}
    print(f"chunk={size:4d}  " + "  ".join(f"recall@{kk}={v/n:.2f}" for kk, v in hits_at.items()))

(HERE / "results").mkdir(exist_ok=True)
json.dump(report, open(HERE / "results" / "retrieval_eval.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
