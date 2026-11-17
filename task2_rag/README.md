# Task 2: RAG from scratch

**Stack:** `all-MiniLM-L6-v2` embeddings + FAISS `IndexFlatIP` (cosine) + NumPy. Optional answer step through a local Ollama model. No LangChain/LlamaIndex.
**Document:** TODO: name + link of the public PDF (≥30 pages). Put it at `data/doc.pdf` (not committed; link it here instead).

## Run
```bash
python rag.py build --pdf data/doc.pdf --chunk 128 --overlap 16
python rag.py build --pdf data/doc.pdf --chunk 512 --overlap 64
python rag.py ask --chunk 128 --q "your question" --llm
python eval_retrieval.py --sizes 128 512 --k 5
```

## Test set
`questions.json`: 15 questions I wrote by reading the PDF. Each one has its page, the full source passage, and a short exact `answer_span` used for automatic hit checking.

## Results (from `results/retrieval_eval.json`)
| Chunk size (words) | # chunks | recall@1 | recall@3 | recall@5 |
|---|---|---|---|---|
| 128 | | | | |
| 512 | | | | |

**Which worked better and why:** TODO (in your own words)

## 2 retrieval failures (questions where `hit_rank` is null)
1. Question / what was retrieved instead / why it failed / what would fix it: TODO
2. TODO

## Concepts (in your own words)
- **What BM25 adds to embedding search:** TODO
- **What a re-ranker does:** TODO
