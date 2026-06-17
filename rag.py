#!/usr/bin/env python3
"""
Offline RAG for Ryzen 7 + Radeon 780M (gfx1103).

Inference and embeddings run through a local Ollama server, which offloads
Gemma 3 4B onto the 780M iGPU via Vulkan (most reliable) or ROCm. Everything
else -- file reading, chunking, the vector index, retrieval -- is pure Python.

Commands:
    python rag.py ingest <file|dir> [<file|dir> ...]
    python rag.py query "your question"
    python rag.py chat
    python rag.py stats
    python rag.py reset

No data leaves the machine.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import sys
from pathlib import Path

import numpy as np

try:
    import faiss
except ImportError:
    sys.exit("Missing dependency. Install with:  pip install faiss-cpu")

try:
    import ollama
    from ollama import Client
except ImportError:
    sys.exit("Missing dependency. Install with:  pip install ollama")


# --------------------------------------------------------------------------
# configuration defaults (override on the command line)
# --------------------------------------------------------------------------

DEFAULTS = {
    "llm_model": "gemma3:4b",
    "embed_model": "nomic-embed-text",
    "store": "rag_store",
    "host": "http://localhost:11434",
    "top_k": 5,
    "chunk_size": 1000,     # characters per chunk
    "chunk_overlap": 150,   # characters carried between adjacent chunks
    "num_ctx": 4096,        # context window; smaller = less KV-cache RAM
    "temperature": 0.2,     # low = grounded, factual answers
    "keep_alive": "5m",     # how long Ollama holds the model after a call.
                            # "0" unloads immediately (frees RAM the moment a
                            # query finishes); "5m" keeps it warm for a session.
}

SUPPORTED_TEXT = {
    ".txt", ".md", ".markdown", ".rst", ".log",
    ".py", ".c", ".cpp", ".h", ".hpp", ".cc", ".cxx",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".csv", ".html", ".xml", ".js", ".ts", ".sh", ".tex",
}
SUPPORTED_PDF = {".pdf"}
SUPPORTED_DOCX = {".docx"}
SUPPORTED = SUPPORTED_TEXT | SUPPORTED_PDF | SUPPORTED_DOCX


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def _get(obj, key, default=None):
    """Read a field from an Ollama response (works whether it's a pydantic
    model or a plain dict)."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def normalize(mat: np.ndarray) -> np.ndarray:
    """L2-normalize rows so inner product == cosine similarity."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


# --------------------------------------------------------------------------
# document reading
# --------------------------------------------------------------------------

def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("PDF support needs:  pip install pypdf")
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def read_docx(path: Path) -> str:
    try:
        import docx
    except ImportError:
        raise ImportError("DOCX support needs:  pip install python-docx")
    document = docx.Document(str(path))
    return "\n".join(p.text for p in document.paragraphs)


def extract(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext in SUPPORTED_PDF:
        return read_pdf(path)
    if ext in SUPPORTED_DOCX:
        return read_docx(path)
    if ext in SUPPORTED_TEXT:
        return read_text_file(path)
    return None


def collect_files(paths) -> list[Path]:
    found: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix.lower() in SUPPORTED:
                    found.append(f)
        elif p.is_file() and p.suffix.lower() in SUPPORTED:
            found.append(p)
        else:
            print(f"  skip (not found / unsupported): {p}")
    return found


# --------------------------------------------------------------------------
# chunking
# --------------------------------------------------------------------------

def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= size:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
                current = ""
            if len(para) <= size:
                current = para
            else:  # paragraph longer than a whole chunk -> hard split
                step = max(1, size - overlap)
                for j in range(0, len(para), step):
                    chunks.append(para[j:j + size])
    if current:
        chunks.append(current)

    # stitch a little context from the previous chunk onto each chunk
    if overlap > 0 and len(chunks) > 1:
        stitched = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-overlap:]
            stitched.append((tail + "\n" + chunks[i]).strip())
        chunks = stitched

    return chunks


# --------------------------------------------------------------------------
# vector store (FAISS flat index, rebuilt from records on load)
# --------------------------------------------------------------------------

class VectorStore:
    """Stores chunk text + its embedding. The FAISS index is just a cache
    rebuilt from the stored vectors, so there is no separate index file to
    keep in sync and updates/removals are trivial."""

    def __init__(self, path: str, embed_model: str):
        self.path = Path(path)
        self.embed_model = embed_model
        self.dim: int | None = None
        self.records: dict[int, dict] = {}        # id -> {text, source, chunk, vec}
        self.files: dict[str, dict] = {}          # abspath -> {hash, ids}
        self.next_id = 0
        self.index = None
        self.order: list[int] = []

    @property
    def meta_file(self) -> Path:
        return self.path / "meta.pkl"

    def load(self) -> "VectorStore":
        if self.meta_file.exists():
            meta = pickle.loads(self.meta_file.read_bytes())
            self.dim = meta["dim"]
            self.embed_model = meta["embed_model"]
            self.records = meta["records"]
            self.files = meta["files"]
            self.next_id = meta["next_id"]
        return self

    def save(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        meta = {
            "dim": self.dim,
            "embed_model": self.embed_model,
            "records": self.records,
            "files": self.files,
            "next_id": self.next_id,
        }
        self.meta_file.write_bytes(pickle.dumps(meta))

    def build_index(self) -> None:
        self.order = list(self.records.keys())
        if not self.order:
            self.index = None
            return
        mat = np.vstack([self.records[i]["vec"] for i in self.order]).astype(np.float32)
        self.index = faiss.IndexFlatIP(mat.shape[1])
        self.index.add(mat)

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[dict, float]]:
        if self.index is None or not self.order:
            return []
        scores, positions = self.index.search(query_vec, min(k, len(self.order)))
        results = []
        for pos, score in zip(positions[0], scores[0]):
            if pos == -1:
                continue
            results.append((self.records[self.order[pos]], float(score)))
        return results


# --------------------------------------------------------------------------
# Ollama interaction
# --------------------------------------------------------------------------

def make_client(args) -> Client:
    return Client(host=args.host, timeout=600)


def ensure_models(client: Client, needed: list[str], host: str) -> None:
    try:
        listing = client.list()
    except Exception as e:
        sys.exit(f"Cannot reach Ollama at {host}. Start it with 'ollama serve'.\n  ({e})")

    models = _get(listing, "models", []) or []
    present: set[str] = set()
    for m in models:
        name = _get(m, "model") or _get(m, "name")
        if name:
            present.add(name)
            present.add(name.split(":")[0])

    missing = [n for n in needed if n not in present and n.split(":")[0] not in present]
    if missing:
        print("Required Ollama models are not installed. Pull them first:")
        for m in missing:
            print(f"    ollama pull {m}")
        sys.exit(1)


def embed_texts(client: Client, model: str, texts: list[str],
                keep_alive: str = "5m", batch: int = 32) -> np.ndarray:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch):
        resp = client.embed(model=model, input=texts[i:i + batch],
                            keep_alive=keep_alive)
        embs = _get(resp, "embeddings")
        if embs is None:
            sys.exit("Embedding response had no 'embeddings' field; check the embed model.")
        vectors.extend(embs)
    return normalize(np.array(vectors, dtype=np.float32))


SYSTEM_PROMPT = (
    "You are a precise offline assistant. Answer the user's question using ONLY "
    "the provided context. If the answer is not contained in the context, say you "
    "don't know rather than guessing. When you use a passage, cite its source "
    "filename in square brackets, e.g. [notes.pdf]."
)


def build_user_prompt(question: str, hits: list[tuple[dict, float]]) -> str:
    if not hits:
        context = "(no relevant context was retrieved)"
    else:
        blocks = []
        for i, (rec, _score) in enumerate(hits, 1):
            src = os.path.basename(rec["source"])
            blocks.append(f"[{i}] source: {src}\n{rec['text']}")
        context = "\n\n".join(blocks)
    return f"Context:\n{context}\n\nQuestion: {question}"


def answer(client, args, question, hits, history=None) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages += history
    messages.append({"role": "user", "content": build_user_prompt(question, hits)})

    options = {"temperature": args.temperature, "num_ctx": args.num_ctx}
    full = ""
    for part in client.chat(model=args.llm_model, messages=messages,
                            options=options, keep_alive=args.keep_alive,
                            stream=True):
        token = _get(_get(part, "message"), "content", "") or ""
        full += token
        print(token, end="", flush=True)
    print()

    if hits:
        sources = sorted({os.path.basename(r["source"]) for r, _ in hits})
        print("\nSources: " + ", ".join(sources))
    return full


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_ingest(args):
    client = make_client(args)
    ensure_models(client, [args.embed_model], args.host)

    store = VectorStore(args.store, args.embed_model).load()
    # keep one embedding space per store
    if store.records and store.embed_model != args.embed_model:
        sys.exit(f"Store was built with embed model '{store.embed_model}'. "
                 f"Use --embed-model {store.embed_model} or 'reset' first.")

    files = collect_files(args.paths)
    if not files:
        print("No supported files found.")
        return

    for f in files:
        key = str(f.resolve())
        h = file_hash(f)

        if key in store.files and store.files[key]["hash"] == h:
            print(f"  unchanged: {f}")
            continue
        if key in store.files:                          # changed -> drop old chunks
            for rid in store.files[key]["ids"]:
                store.records.pop(rid, None)
            print(f"  updating:  {f}")
        else:
            print(f"  adding:    {f}")

        try:
            raw = extract(f)
        except ImportError as e:
            print(f"    ! {e}")
            continue
        except Exception as e:
            print(f"    ! could not read ({e})")
            continue

        if not raw or not raw.strip():
            print("    (no extractable text)")
            store.files[key] = {"hash": h, "ids": []}
            continue

        chunks = chunk_text(raw, args.chunk_size, args.chunk_overlap)
        vecs = embed_texts(client, args.embed_model, chunks, args.keep_alive)

        ids = []
        for ci, (chunk, vec) in enumerate(zip(chunks, vecs)):
            rid = store.next_id
            store.next_id += 1
            store.records[rid] = {"text": chunk, "source": str(f),
                                  "chunk": ci, "vec": vec}
            ids.append(rid)
        store.files[key] = {"hash": h, "ids": ids}
        store.dim = int(vecs.shape[1])
        print(f"    {len(chunks)} chunks")

    store.save()
    print(f"\nDone. {len(store.records)} chunks across "
          f"{len(store.files)} files in '{args.store}'.")


def cmd_query(args):
    store = VectorStore(args.store, args.embed_model).load()
    if not store.records:
        print("Index is empty. Run:  python rag.py ingest <files>")
        return
    embed_model = store.embed_model or args.embed_model

    client = make_client(args)
    ensure_models(client, [embed_model, args.llm_model], args.host)

    store.build_index()
    qvec = embed_texts(client, embed_model, [args.question], args.keep_alive)
    hits = store.search(qvec, args.top_k)
    answer(client, args, args.question, hits)


def cmd_chat(args):
    store = VectorStore(args.store, args.embed_model).load()
    if not store.records:
        print("Index is empty. Run:  python rag.py ingest <files>")
        return
    embed_model = store.embed_model or args.embed_model

    client = make_client(args)
    ensure_models(client, [embed_model, args.llm_model], args.host)
    store.build_index()

    print("Offline RAG chat  (type 'exit' or Ctrl-D to quit)\n")
    history: list[dict] = []
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        qvec = embed_texts(client, embed_model, [question], args.keep_alive)
        hits = store.search(qvec, args.top_k)
        print("rag> ", end="", flush=True)
        reply = answer(client, args, question, hits, history)

        history += [{"role": "user", "content": question},
                    {"role": "assistant", "content": reply}]
        history = history[-6:]      # remember the last 3 turns
        print()


def cmd_stats(args):
    store = VectorStore(args.store, args.embed_model).load()
    print(f"Store:       {args.store}")
    print(f"Embed model: {store.embed_model}")
    print(f"Dimension:   {store.dim}")
    print(f"Files:       {len(store.files)}   Chunks: {len(store.records)}")
    for path, info in sorted(store.files.items()):
        print(f"  {len(info['ids']):5d} chunks   {path}")


def cmd_reset(args):
    target = Path(args.store) / "meta.pkl"
    target.unlink(missing_ok=True)
    print(f"Cleared '{args.store}'.")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--store", default=DEFAULTS["store"],
                        help="directory holding the index (default: %(default)s)")
    common.add_argument("--host", default=DEFAULTS["host"],
                        help="Ollama server URL (default: %(default)s)")
    common.add_argument("--llm-model", default=DEFAULTS["llm_model"],
                        help="generation model (default: %(default)s)")
    common.add_argument("--embed-model", default=DEFAULTS["embed_model"],
                        help="embedding model (default: %(default)s)")
    common.add_argument("--top-k", type=int, default=DEFAULTS["top_k"],
                        help="chunks to retrieve (default: %(default)s)")
    common.add_argument("--chunk-size", type=int, default=DEFAULTS["chunk_size"])
    common.add_argument("--chunk-overlap", type=int, default=DEFAULTS["chunk_overlap"])
    common.add_argument("--num-ctx", type=int, default=DEFAULTS["num_ctx"])
    common.add_argument("--temperature", type=float, default=DEFAULTS["temperature"])
    common.add_argument("--keep-alive", default=DEFAULTS["keep_alive"],
                        help="how long Ollama holds the model after a call; "
                             "'0' frees RAM immediately (default: %(default)s)")

    parser = argparse.ArgumentParser(
        description="Offline RAG (Gemma 3 4B via Ollama) for Ryzen 7 + Radeon 780M.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", parents=[common], help="vectorize files/folders")
    p.add_argument("paths", nargs="+")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("query", parents=[common], help="ask one question")
    p.add_argument("question")
    p.set_defaults(func=cmd_query)

    sub.add_parser("chat", parents=[common], help="interactive session").set_defaults(func=cmd_chat)
    sub.add_parser("stats", parents=[common], help="show what's indexed").set_defaults(func=cmd_stats)
    sub.add_parser("reset", parents=[common], help="wipe the index").set_defaults(func=cmd_reset)
    return parser


def main():
    # Windows consoles default to a legacy codepage; force UTF-8 so streamed
    # model tokens (em-dashes, accents, etc.) don't raise UnicodeEncodeError.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()