# LLM Engineering Assignment

Three tasks, one folder each. All numbers in `report/` come from my own runs (see each task's `results/` folder).

| Folder | Task | Run on |
|---|---|---|
| `task1_lora/` | QLoRA fine-tune of Qwen2.5-0.5B-Instruct: plain-English request → fixed JSON | Colab / Kaggle T4 |
| `task2_rag/` | RAG from scratch (sentence-transformers + FAISS, no frameworks) | CPU or Colab |
| `task3_local/` | Local inference with Ollama, two quantisation levels, speed/memory/quality | Own laptop |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Each task folder has its own `README.md` with the exact commands, run in this order.

## Report
`report/REPORT.md` → export to PDF (max 4 pages) → `report/report.pdf`.
