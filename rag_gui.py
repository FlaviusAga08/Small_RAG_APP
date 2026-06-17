#!/usr/bin/env python3
"""
Windowed front-end for the offline RAG engine (rag.py).

Double-clickable app: add documents, ask questions, get grounded answers from
a local Gemma 3 4B running on the 780M via Ollama. All work runs in background
threads so the window never freezes, and the index is stored per-user under
%LOCALAPPDATA%\\OfflineRAG so it survives and needs no admin rights.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
from pathlib import Path

# In a PyInstaller --windowed build there is no console, so stdout/stderr are
# None. Any stray print() would then crash; route them to nowhere.
for _name in ("stdout", "stderr"):
    if getattr(sys, _name) is None:
        setattr(sys, _name, open(os.devnull, "w"))

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import rag
from ollama import Client


APP_NAME = "OfflineRAG"


def appdata_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    d = Path(base) / APP_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


CFG = {
    "host": rag.DEFAULTS["host"],
    "llm_model": rag.DEFAULTS["llm_model"],
    "embed_model": rag.DEFAULTS["embed_model"],
    "store": str(appdata_dir() / "rag_store"),
    "top_k": rag.DEFAULTS["top_k"],
    "num_ctx": rag.DEFAULTS["num_ctx"],
    "temperature": rag.DEFAULTS["temperature"],
    "keep_alive": rag.DEFAULTS["keep_alive"],
    "chunk_size": rag.DEFAULTS["chunk_size"],
    "chunk_overlap": rag.DEFAULTS["chunk_overlap"],
}


class RagApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.events: queue.Queue = queue.Queue()
        self.busy = False
        self.client = Client(host=CFG["host"], timeout=600)
        self.store = rag.VectorStore(CFG["store"], CFG["embed_model"]).load()

        self._build_ui()
        self.root.after(50, self._drain)
        # first-run checks (Ollama reachable, models present) off the UI thread
        self._start(self._startup)

    # ---------------------------- UI ----------------------------

    def _build_ui(self):
        self.root.minsize(640, 480)
        try:
            ttk.Style().theme_use("vista")
        except tk.TclError:
            pass

        bar = ttk.Frame(self.root, padding=(10, 8))
        bar.pack(fill="x")
        self.btn_files = ttk.Button(bar, text="Add files…", command=self.on_add_files)
        self.btn_files.pack(side="left")
        self.btn_folder = ttk.Button(bar, text="Add folder…", command=self.on_add_folder)
        self.btn_folder.pack(side="left", padx=(6, 0))
        self.btn_reset = ttk.Button(bar, text="Forget all", command=self.on_reset)
        self.btn_reset.pack(side="left", padx=(6, 0))

        self.output = scrolledtext.ScrolledText(
            self.root, wrap="word", state="disabled",
            font=("Segoe UI", 10), padx=10, pady=8)
        self.output.pack(fill="both", expand=True, padx=10)
        self.output.tag_config("muted", foreground="#777777")
        self.output.tag_config("you", foreground="#1a5fb4", font=("Segoe UI", 10, "bold"))
        self.output.tag_config("bot", foreground="#000000", font=("Segoe UI", 10, "bold"))
        self.output.tag_config("src", foreground="#777777", font=("Segoe UI", 9, "italic"))

        row = ttk.Frame(self.root, padding=(10, 8))
        row.pack(fill="x")
        self.entry = ttk.Entry(row, font=("Segoe UI", 10))
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.bind("<Return>", self.on_ask)
        self.btn_ask = ttk.Button(row, text="Ask", command=self.on_ask)
        self.btn_ask.pack(side="left", padx=(6, 0))

        self.status = ttk.Label(self.root, text="Starting…", anchor="w",
                                relief="sunken", padding=(8, 3))
        self.status.pack(fill="x", side="bottom")

    def _write(self, text: str, tag: str | None = None):
        self.output.configure(state="normal")
        self.output.insert("end", text, tag or ())
        self.output.see("end")
        self.output.configure(state="disabled")

    def _set_inputs(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        for w in (self.btn_files, self.btn_folder, self.btn_reset,
                  self.btn_ask, self.entry):
            w.configure(state=state)

    # ------------------------ event pump ------------------------

    def _drain(self):
        try:
            while True:
                kind, *payload = self.events.get_nowait()
                if kind == "status":
                    self.status.config(text=payload[0])
                elif kind == "log":
                    self._write(payload[0] + "\n", "muted")
                elif kind == "you":
                    self._write("\nYou: ", "you")
                    self._write(payload[0] + "\n")
                elif kind == "bot_start":
                    self._write("Assistant: ", "bot")
                elif kind == "token":
                    self._write(payload[0])
                elif kind == "bot_end":
                    srcs = payload[0]
                    self._write("\n")
                    if srcs:
                        self._write("sources: " + ", ".join(srcs) + "\n", "src")
                elif kind == "busy_done":
                    self.busy = False
                    self._set_inputs(True)
        except queue.Empty:
            pass
        self.root.after(50, self._drain)

    # ------------------------ worker glue -----------------------

    def _start(self, fn, *args):
        if self.busy:
            return
        self.busy = True
        self._set_inputs(False)

        def wrap():
            try:
                fn(*args)
            except Exception as e:
                self.events.put(("log", f"Error: {e}"))
                self.events.put(("status", "Error — see message above."))
            finally:
                self.events.put(("busy_done",))

        threading.Thread(target=wrap, daemon=True).start()

    # ------------------------- workers --------------------------

    def _startup(self):
        try:
            listing = self.client.list()
        except Exception:
            self.events.put(("status", "Ollama is not running."))
            self.events.put(("log",
                "Could not reach Ollama. Make sure the Ollama tray app is "
                "running, then reopen this app."))
            return

        present = set()
        for m in (rag._get(listing, "models", []) or []):
            name = rag._get(m, "model") or rag._get(m, "name")
            if name:
                present.add(name)
                present.add(name.split(":")[0])

        for model in (CFG["embed_model"], CFG["llm_model"]):
            if model not in present and model.split(":")[0] not in present:
                self._pull(model)

        n = len(self.store.records)
        if n:
            self.store.build_index()
            self.events.put(("status",
                f"Ready — {n} chunks from {len(self.store.files)} files. Ask away."))
        else:
            self.events.put(("status", "Ready — add documents to get started."))

    def _pull(self, model: str):
        self.events.put(("log", f"Downloading model '{model}' (one-time, first run)…"))
        last = -1
        for prog in self.client.pull(model, stream=True):
            total = rag._get(prog, "total") or 0
            done = rag._get(prog, "completed") or 0
            if total:
                pct = int(done * 100 / total)
                if pct != last:
                    last = pct
                    self.events.put(("status", f"Downloading {model}: {pct}%"))
        self.events.put(("status", f"{model} ready."))

    def _ingest(self, paths):
        files = rag.collect_files(paths)
        if not files:
            self.events.put(("status", "No supported files found."))
            return
        for f in files:
            key = str(f.resolve())
            h = rag.file_hash(f)
            if key in self.store.files and self.store.files[key]["hash"] == h:
                continue
            if key in self.store.files:
                for rid in self.store.files[key]["ids"]:
                    self.store.records.pop(rid, None)

            self.events.put(("status", f"Reading {f.name} …"))
            try:
                raw = rag.extract(f)
            except Exception as e:
                self.events.put(("log", f"Skipped {f.name}: {e}"))
                continue
            if not raw or not raw.strip():
                self.store.files[key] = {"hash": h, "ids": []}
                continue

            chunks = rag.chunk_text(raw, CFG["chunk_size"], CFG["chunk_overlap"])
            self.events.put(("status", f"Indexing {f.name} ({len(chunks)} chunks) …"))
            vecs = rag.embed_texts(self.client, CFG["embed_model"],
                                   chunks, CFG["keep_alive"])
            ids = []
            for ci, (ch, v) in enumerate(zip(chunks, vecs)):
                rid = self.store.next_id
                self.store.next_id += 1
                self.store.records[rid] = {"text": ch, "source": str(f),
                                           "chunk": ci, "vec": v}
                ids.append(rid)
            self.store.files[key] = {"hash": h, "ids": ids}
            self.store.dim = int(vecs.shape[1])
            self.events.put(("log", f"Added {f.name}"))

        self.store.save()
        self.store.build_index()
        self.events.put(("status",
            f"Done — {len(self.store.records)} chunks from {len(self.store.files)} files."))

    def _ask(self, question: str):
        if not self.store.records:
            self.events.put(("log", "No documents indexed yet — add some first."))
            return
        self.store.build_index()
        qvec = rag.embed_texts(self.client, CFG["embed_model"],
                               [question], CFG["keep_alive"])
        hits = self.store.search(qvec, CFG["top_k"])

        messages = [
            {"role": "system", "content": rag.SYSTEM_PROMPT},
            {"role": "user", "content": rag.build_user_prompt(question, hits)},
        ]
        options = {"temperature": CFG["temperature"], "num_ctx": CFG["num_ctx"]}
        self.events.put(("bot_start",))
        for part in self.client.chat(model=CFG["llm_model"], messages=messages,
                                     options=options, keep_alive=CFG["keep_alive"],
                                     stream=True):
            token = rag._get(rag._get(part, "message"), "content", "") or ""
            if token:
                self.events.put(("token", token))
        srcs = sorted({os.path.basename(r["source"]) for r, _ in hits}) if hits else []
        self.events.put(("bot_end", srcs))

    def _reset(self):
        (Path(CFG["store"]) / "meta.pkl").unlink(missing_ok=True)
        self.store = rag.VectorStore(CFG["store"], CFG["embed_model"]).load()
        self.events.put(("status", "Index cleared — add documents to get started."))

    # ----------------------- UI callbacks -----------------------

    def on_add_files(self):
        paths = filedialog.askopenfilenames(title="Choose documents")
        if paths:
            self._start(self._ingest, list(paths))

    def on_add_folder(self):
        folder = filedialog.askdirectory(title="Choose a folder")
        if folder:
            self._start(self._ingest, [folder])

    def on_reset(self):
        if messagebox.askyesno("Forget all", "Remove all indexed documents?"):
            self._start(self._reset)

    def on_ask(self, event=None):
        q = self.entry.get().strip()
        if not q or self.busy:
            return
        self.entry.delete(0, "end")
        self.events.put(("you", q))
        self._start(self._ask, q)


def main():
    root = tk.Tk()
    root.title("Offline RAG")
    root.geometry("840x640")
    RagApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
