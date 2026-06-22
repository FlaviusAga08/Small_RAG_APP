from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import sys
from pathlib import Path
import fitz
import numpy as np

try:
    import faiss
except ImportError:
    sys.exit("Missing dependency. Install with:  pip install faiss-cpu")

try:
    from openai import OpenAI
except ImportError:
    sys.exit("Missing dependency. Install with:  pip install openai")

try:
    from dotenv import load_dotenv
except ImportError:
    sys.exit("Missing dependency. Install with:  pip install python-dotenv")

load_dotenv()

# --------------------------------------------------------------------------
# configuration defaults (override via .env or command line)
# --------------------------------------------------------------------------

DEFAULTS = {
    "llm_model":     os.getenv("VAST_LLM_MODEL", "llama3"),
    "embed_model":   os.getenv("VAST_EMBED_MODEL", "nomic-embed-text"),
    "store":         "rag_store",
    "base_url":      os.getenv("VAST_BASE_URL", "http://localhost:8000/v1"),
    "top_k":         10,
    "chunk_size":    2000,     # characters per chunk
    "chunk_overlap": 400,      # characters carried between adjacent chunks
    "num_ctx":       4096,     # max tokens in the LLM response
    "temperature":   0.2,      # low = grounded, factual answers
}

SUPPORTED_TEXT = {
    ".txt", ".md", ".markdown", ".rst", ".log",
    ".py", ".c", ".cpp", ".h", ".hpp", ".cc", ".cxx",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".csv", ".html", ".xml", ".js", ".ts", ".sh", ".tex",
}
SUPPORTED_PDF  = {".pdf"}
SUPPORTED_DOCX = {".docx"}
SUPPORTED = SUPPORTED_TEXT | SUPPORTED_PDF | SUPPORTED_DOCX


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def _get(obj, key, default=None):
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
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)


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
# vast.ai / OpenAI-compatible API interaction
# --------------------------------------------------------------------------

def make_client(args) -> OpenAI:
    api_key = os.getenv("VAST_API_KEY", "not-required")
    return OpenAI(base_url=args.base_url, api_key=api_key, timeout=600)


def check_connection(client: OpenAI, base_url: str) -> None:
    try:
        client.models.list()
    except Exception as e:
        sys.exit(
            f"Cannot reach API at {base_url}.\n"
            f"Check VAST_BASE_URL and VAST_API_KEY in your .env file.\n  ({e})"
        )


def embed_texts(client: OpenAI, model: str, texts: list[str],
                batch: int = 32) -> np.ndarray:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch):
        resp = client.embeddings.create(model=model, input=texts[i:i + batch])
        vectors.extend(item.embedding for item in resp.data)
    return normalize(np.array(vectors, dtype=np.float32))


SYSTEM_PROMPT = """
Ești un asistent RAG.

Primești:
1. Un context extras din documente.
2. O întrebare.

Trebuie să răspunzi doar pe baza contextului.

Dacă răspunsul apare în context, citează și rezumă informația relevantă.

Dacă informația nu apare în context, răspunde exact:
"Nu am găsit această informație în documentele furnizate."

Nu spune că informația lipsește dacă există în context.
Nu face presupuneri.
Nu adăuga cunoștințe proprii.
"""

def build_user_prompt(question: str, hits: list[tuple[dict, float]]) -> str:
    if not hits:
        context = "(no relevant context was retrieved)"
    else:
        blocks = []
        for i, (rec, _score) in enumerate(hits, 1):
            src = os.path.basename(rec["source"])
            blocks.append(f"[{i}] source: {src}\n{rec['text']}")
        context = "\n\n".join(blocks)
    return f"""
        Context:

        {context}

        Întrebare:
        {question}

        Răspuns:
        """


def answer(client: OpenAI, args, question: str,
           hits: list[tuple[dict, float]], history=None) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages += history
    messages.append({"role": "user", "content": build_user_prompt(question, hits)})

    full = ""
    stream = client.chat.completions.create(
        model=args.llm_model,
        messages=messages,
        temperature=args.temperature,
        max_tokens=args.num_ctx,
        stream=True,
    )
    for chunk in stream:
        token = chunk.choices[0].delta.content or ""
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
    check_connection(client, args.base_url)

    store = VectorStore(args.store, args.embed_model).load()
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
        if key in store.files:
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
        vecs = embed_texts(client, args.embed_model, chunks)

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
    check_connection(client, args.base_url)

    store.build_index()
    qvec = embed_texts(client, embed_model, [args.question])
    hits = store.search(qvec, args.top_k)
    answer(client, args, args.question, hits)


def cmd_chat(args):
    store = VectorStore(args.store, args.embed_model).load()
    if not store.records:
        print("Index is empty. Run:  python rag.py ingest <files>")
        return
    embed_model = store.embed_model or args.embed_model

    client = make_client(args)
    check_connection(client, args.base_url)
    store.build_index()

    print("RAG chat  (type 'exit' or Ctrl-D to quit)\n")
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

        qvec = embed_texts(client, embed_model, [question])
        hits = store.search(qvec, args.top_k)

        for i, (rec, score) in enumerate(hits):
            print(f"\n=== HIT {i+1} ===")
            print(f"score={score:.4f}")
            print(rec["text"][:3000])
            print()
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
    common.add_argument("--base-url", default=DEFAULTS["base_url"],
                        help="OpenAI-compatible API base URL, e.g. "
                             "http://your-instance.vast.ai:port/v1 "
                             "(default: %(default)s, override with VAST_BASE_URL in .env)")
    common.add_argument("--llm-model", default=DEFAULTS["llm_model"],
                        help="generation model name on the remote server (default: %(default)s)")
    common.add_argument("--embed-model", default=DEFAULTS["embed_model"],
                        help="embedding model name on the remote server (default: %(default)s)")
    common.add_argument("--top-k", type=int, default=DEFAULTS["top_k"],
                        help="chunks to retrieve (default: %(default)s)")
    common.add_argument("--chunk-size", type=int, default=DEFAULTS["chunk_size"])
    common.add_argument("--chunk-overlap", type=int, default=DEFAULTS["chunk_overlap"])
    common.add_argument("--num-ctx", type=int, default=DEFAULTS["num_ctx"],
                        help="max tokens in the LLM response (default: %(default)s)")
    common.add_argument("--temperature", type=float, default=DEFAULTS["temperature"])

    parser = argparse.ArgumentParser(
        description="RAG powered by a remote LLM on vast.ai (OpenAI-compatible API).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", parents=[common], help="vectorize files/folders")
    p.add_argument("paths", nargs="+")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("query", parents=[common], help="ask one question")
    p.add_argument("question")
    p.set_defaults(func=cmd_query)

    sub.add_parser("chat",  parents=[common], help="interactive session").set_defaults(func=cmd_chat)
    sub.add_parser("stats", parents=[common], help="show what's indexed").set_defaults(func=cmd_stats)
    sub.add_parser("reset", parents=[common], help="wipe the index").set_defaults(func=cmd_reset)
    return parser


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
