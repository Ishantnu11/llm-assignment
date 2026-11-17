"""
RAG from scratch: chunk -> embed -> FAISS -> retrieve -> (optional) LLM answer.
No LangChain / LlamaIndex.

    python rag.py build --pdf data/doc.pdf --chunk 256 --overlap 32
    python rag.py ask   --chunk 256 --q "What is ...?"            # retrieval only
    python rag.py ask   --chunk 256 --q "What is ...?" --llm      # + answer via Ollama

Chunk size is counted in words (simple, easy to explain).
"""
import argparse, json, os
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).parent
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_model = None


def embedder():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMB_MODEL)
    return _model


def load_pages(pdf_path):
    reader = PdfReader(pdf_path)
    return [(i + 1, (p.extract_text() or "")) for i, p in enumerate(reader.pages)]


def chunk_pages(pages, size, overlap):
    """Sliding window over words; each chunk remembers the page it starts on."""
    words = []  # (word, page)
    for page_no, text in pages:
        words += [(w, page_no) for w in text.split()]
    chunks, step = [], size - overlap
    for start in range(0, len(words), step):
        win = words[start:start + size]
        if not win:
            break
        chunks.append({"id": len(chunks), "page": win[0][1],
                       "text": " ".join(w for w, _ in win)})
        if start + size >= len(words):
            break
    return chunks


def index_dir(size):
    return HERE / "index" / f"chunk{size}"


def build(pdf, size, overlap):
    pages = load_pages(pdf)
    print(f"{len(pages)} pages")
    chunks = chunk_pages(pages, size, overlap)
    vecs = embedder().encode([c["text"] for c in chunks], batch_size=64,
                             normalize_embeddings=True, show_progress_bar=True)
    vecs = np.asarray(vecs, dtype="float32")
    idx = faiss.IndexFlatIP(vecs.shape[1])  # inner product on unit vectors = cosine
    idx.add(vecs)
    d = index_dir(size)
    d.mkdir(parents=True, exist_ok=True)
    faiss.write_index(idx, str(d / "faiss.index"))
    json.dump(chunks, open(d / "chunks.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(f"built {len(chunks)} chunks of {size} words -> {d}")


def load(size):
    d = index_dir(size)
    return faiss.read_index(str(d / "faiss.index")), json.load(open(d / "chunks.json", encoding="utf-8"))


def search(query, size, k=5, _cache={}):
    if size not in _cache:
        _cache[size] = load(size)
    idx, chunks = _cache[size]
    q = embedder().encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = idx.search(q, k)
    return [(chunks[i], float(s)) for i, s in zip(ids[0], scores[0])]


def answer(query, hits, model=os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")):
    import requests
    ctx = "\n\n".join(f"[p.{c['page']}] {c['text']}" for c, _ in hits)
    prompt = (f"Answer the question using only the context. Cite page numbers. "
              f"If the answer is not in the context, say so.\n\nContext:\n{ctx}\n\nQuestion: {query}\nAnswer:")
    r = requests.post("http://localhost:11434/api/generate",
                      json={"model": model, "prompt": prompt, "stream": False}, timeout=300)
    return r.json()["response"]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--pdf", required=True)
    b.add_argument("--chunk", type=int, default=256); b.add_argument("--overlap", type=int, default=32)
    a = sub.add_parser("ask"); a.add_argument("--q", required=True)
    a.add_argument("--chunk", type=int, default=256); a.add_argument("--k", type=int, default=5)
    a.add_argument("--llm", action="store_true")
    args = p.parse_args()

    if args.cmd == "build":
        build(args.pdf, args.chunk, args.overlap)
    else:
        hits = search(args.q, args.chunk, args.k)
        for c, s in hits:
            print(f"[{s:.3f}] p.{c['page']}: {c['text'][:160]}...")
        if args.llm:
            print("\n" + answer(args.q, hits))
