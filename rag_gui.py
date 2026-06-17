#!/usr/bin/env python3
"""
Modern desktop GUI for the offline RAG engine (rag.py).
Built with CustomTkinter — clean dark/light mode, rounded corners,
proper typography. Works on Windows, Linux, macOS.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
from pathlib import Path

for _name in ("stdout", "stderr"):
    if getattr(sys, _name) is None:
        setattr(sys, _name, open(os.devnull, "w"))

import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk

import rag
from ollama import Client

APP_NAME  = "OfflineRAG"
APP_TITLE = "Offline RAG"
W, H      = 1000, 700

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

ACCENT   = "#7C6FF7"
ACCENT_H = "#9D98F5"
BG_MAIN  = "#1A1A2E"
BG_SIDE  = "#16213E"
BG_CARD  = "#0F3460"
BG_INPUT = "#1E2A45"
BG_MSG_U = "#2D2D5E"
BG_MSG_A = "#1E3A5F"
TEXT_PRI = "#E8E8F0"
TEXT_SEC = "#9090A8"
TEXT_ACC = "#C4C0FF"
BORDER   = "#2A2A4A"
SUCCESS  = "#4CAF8A"
WARNING  = "#F0A830"
DANGER   = "#E05555"


def appdata_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    d = Path(base) / APP_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


CFG = {
    "host":         rag.DEFAULTS["host"],
    "llm_model":    rag.DEFAULTS["llm_model"],
    "embed_model":  rag.DEFAULTS["embed_model"],
    "store":        str(appdata_dir() / "rag_store"),
    "top_k":        rag.DEFAULTS["top_k"],
    "num_ctx":      rag.DEFAULTS["num_ctx"],
    "temperature":  rag.DEFAULTS["temperature"],
    "keep_alive":   rag.DEFAULTS["keep_alive"],
    "chunk_size":   rag.DEFAULTS["chunk_size"],
    "chunk_overlap":rag.DEFAULTS["chunk_overlap"],
}


class RagApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry(f"{W}x{H}")
        self.minsize(800, 560)
        self.configure(fg_color=BG_MAIN)

        self.events: queue.Queue = queue.Queue()
        self.busy   = False
        self.client = Client(host=CFG["host"], timeout=600)
        self.store  = rag.VectorStore(CFG["store"], CFG["embed_model"]).load()
        self._doc_vars: list[tuple[str, ctk.StringVar]] = []

        self._build()
        self.after(50, self._drain)
        self._start(self._startup)

    # ─────────────────────────── layout ────────────────────────────

    def _build(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ── sidebar ──────────────────────────────────────────────
        side = ctk.CTkFrame(self, fg_color=BG_SIDE, corner_radius=0, width=240)
        side.grid(row=0, column=0, sticky="nsew")
        side.grid_propagate(False)
        side.grid_rowconfigure(4, weight=1)

        logo = ctk.CTkLabel(side, text="⚡ Offline RAG",
                            font=ctk.CTkFont(size=18, weight="bold"),
                            text_color=TEXT_ACC)
        logo.grid(row=0, column=0, padx=20, pady=(24, 4), sticky="w")

        sub = ctk.CTkLabel(side, text="Local AI · No internet",
                           font=ctk.CTkFont(size=11),
                           text_color=TEXT_SEC)
        sub.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="w")

        ctk.CTkLabel(side, text="DOCUMENTS",
                     font=ctk.CTkFont(size=10, weight="bold"),
                     text_color=TEXT_SEC).grid(
            row=2, column=0, padx=20, pady=(0, 6), sticky="w")

        self.doc_scroll = ctk.CTkScrollableFrame(
            side, fg_color="transparent", height=220)
        self.doc_scroll.grid(row=3, column=0, padx=12, pady=0, sticky="ew")
        self._refresh_doc_list()

        btn_frame = ctk.CTkFrame(side, fg_color="transparent")
        btn_frame.grid(row=5, column=0, padx=12, pady=12, sticky="ew")
        btn_frame.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkButton(btn_frame, text="Add files",
                      command=self.on_add_files,
                      fg_color=ACCENT, hover_color=ACCENT_H,
                      font=ctk.CTkFont(size=12),
                      height=34, corner_radius=8).grid(
            row=0, column=0, padx=(0, 4), sticky="ew")

        ctk.CTkButton(btn_frame, text="Add folder",
                      command=self.on_add_folder,
                      fg_color=BG_CARD, hover_color=BORDER,
                      font=ctk.CTkFont(size=12),
                      height=34, corner_radius=8).grid(
            row=0, column=1, padx=(4, 0), sticky="ew")

        ctk.CTkButton(side, text="Clear index",
                      command=self.on_reset,
                      fg_color="transparent",
                      border_width=1, border_color=DANGER,
                      text_color=DANGER,
                      hover_color="#2A1515",
                      font=ctk.CTkFont(size=12),
                      height=32, corner_radius=8).grid(
            row=6, column=0, padx=12, pady=(0, 12), sticky="ew")

        # status pill
        self.status_dot = ctk.CTkLabel(side, text="●", text_color=WARNING,
                                       font=ctk.CTkFont(size=10))
        self.status_dot.grid(row=7, column=0, padx=(20, 4), pady=(0, 20), sticky="w")
        self.status_lbl = ctk.CTkLabel(side, text="Starting…",
                                       font=ctk.CTkFont(size=11),
                                       text_color=TEXT_SEC,
                                       wraplength=180, justify="left")
        self.status_lbl.grid(row=7, column=0, padx=(34, 12), pady=(0, 20), sticky="w")

        # ── main area ────────────────────────────────────────────
        main = ctk.CTkFrame(self, fg_color=BG_MAIN, corner_radius=0)
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_rowconfigure(0, weight=1)
        main.grid_columnconfigure(0, weight=1)

        # chat area
        self.chat = ctk.CTkScrollableFrame(main, fg_color="transparent")
        self.chat.grid(row=0, column=0, sticky="nsew", padx=24, pady=(20, 0))
        self.chat.grid_columnconfigure(0, weight=1)
        self._chat_row = 0

        self._show_welcome()

        # input row
        inp = ctk.CTkFrame(main, fg_color=BG_INPUT,
                           corner_radius=16, height=56)
        inp.grid(row=1, column=0, sticky="ew", padx=24, pady=16)
        inp.grid_columnconfigure(0, weight=1)
        inp.grid_propagate(False)

        self.entry = ctk.CTkEntry(
            inp, placeholder_text="Ask a question about your documents…",
            fg_color="transparent", border_width=0,
            font=ctk.CTkFont(size=13), text_color=TEXT_PRI,
            placeholder_text_color=TEXT_SEC)
        self.entry.grid(row=0, column=0, padx=(16, 8), pady=10, sticky="ew")
        self.entry.bind("<Return>", self.on_ask)

        self.btn_ask = ctk.CTkButton(
            inp, text="Ask →", width=80, height=36,
            fg_color=ACCENT, hover_color=ACCENT_H,
            font=ctk.CTkFont(size=13, weight="bold"),
            corner_radius=10, command=self.on_ask)
        self.btn_ask.grid(row=0, column=1, padx=(0, 10), pady=10)

    # ─────────────────────── chat helpers ───────────────────────

    def _show_welcome(self):
        card = ctk.CTkFrame(self.chat, fg_color=BG_CARD,
                            corner_radius=12)
        card.grid(row=self._chat_row, column=0,
                  sticky="ew", pady=(0, 16))
        self._chat_row += 1
        card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(card, text="Welcome to Offline RAG",
                     font=ctk.CTkFont(size=16, weight="bold"),
                     text_color=TEXT_ACC).grid(
            row=0, column=0, padx=20, pady=(16, 4), sticky="w")
        ctk.CTkLabel(card,
                     text="Add documents on the left, then ask questions here.\n"
                          "Everything runs on this machine — nothing leaves.",
                     font=ctk.CTkFont(size=13),
                     text_color=TEXT_SEC,
                     justify="left").grid(
            row=1, column=0, padx=20, pady=(0, 16), sticky="w")

    def _bubble(self, role: str, text: str = "", sources: list[str] | None = None):
        """Add a chat bubble. Returns the label so tokens can be appended."""
        is_user = role == "user"
        outer = ctk.CTkFrame(self.chat, fg_color="transparent")
        outer.grid(row=self._chat_row, column=0,
                   sticky="e" if is_user else "ew",
                   pady=(0, 12))
        self._chat_row += 1
        outer.grid_columnconfigure(0, weight=1)

        if not is_user:
            tag = ctk.CTkLabel(outer, text="Assistant",
                               font=ctk.CTkFont(size=10, weight="bold"),
                               text_color=TEXT_ACC)
            tag.grid(row=0, column=0, padx=4, pady=(0, 2), sticky="w")

        bubble = ctk.CTkFrame(
            outer,
            fg_color=BG_MSG_U if is_user else BG_MSG_A,
            corner_radius=12)
        bubble.grid(row=1 if not is_user else 0, column=0,
                    sticky="e" if is_user else "ew")
        bubble.grid_columnconfigure(0, weight=1)

        lbl = ctk.CTkLabel(
            bubble, text=text,
            font=ctk.CTkFont(size=13),
            text_color=TEXT_PRI,
            justify="left", wraplength=560,
            anchor="w")
        lbl.grid(row=0, column=0, padx=14, pady=10, sticky="ew")

        if sources:
            src_text = "Sources: " + " · ".join(sources)
            ctk.CTkLabel(bubble, text=src_text,
                         font=ctk.CTkFont(size=10),
                         text_color=TEXT_SEC,
                         justify="left").grid(
                row=1, column=0, padx=14, pady=(0, 8), sticky="w")

        self._scroll_bottom()
        return lbl

    def _scroll_bottom(self):
        self.chat.after(50, lambda: self.chat._parent_canvas.yview_moveto(1.0))

    def _refresh_doc_list(self):
        for w in self.doc_scroll.winfo_children():
            w.destroy()
        self._doc_vars.clear()
        if not self.store.files:
            ctk.CTkLabel(self.doc_scroll, text="No documents yet",
                         font=ctk.CTkFont(size=11),
                         text_color=TEXT_SEC).pack(anchor="w", pady=4)
            return
        for path, info in sorted(self.store.files.items()):
            name = os.path.basename(path)
            n    = len(info["ids"])
            row  = ctk.CTkFrame(self.doc_scroll, fg_color="transparent")
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text="📄",
                         font=ctk.CTkFont(size=11)).pack(side="left")
            ctk.CTkLabel(row,
                         text=name if len(name) < 22 else name[:19] + "…",
                         font=ctk.CTkFont(size=11),
                         text_color=TEXT_PRI).pack(side="left", padx=(4, 0))
            ctk.CTkLabel(row, text=f"{n}c",
                         font=ctk.CTkFont(size=10),
                         text_color=TEXT_SEC).pack(side="right")

    # ─────────────────────── event pump ─────────────────────────

    def _drain(self):
        try:
            while True:
                ev = self.events.get_nowait()
                kind = ev[0]
                if kind == "status":
                    self.status_lbl.configure(text=ev[1])
                elif kind == "dot":
                    self.status_dot.configure(text_color=ev[1])
                elif kind == "log":
                    self._bubble("assistant", ev[1])
                elif kind == "you":
                    self._bubble("user", ev[1])
                elif kind == "bot_start":
                    self._active_lbl = self._bubble("assistant", "")
                elif kind == "token":
                    cur = self._active_lbl.cget("text")
                    self._active_lbl.configure(text=cur + ev[1])
                    self._scroll_bottom()
                elif kind == "bot_end":
                    if ev[1]:
                        parent = self._active_lbl.master
                        ctk.CTkLabel(parent,
                                     text="Sources: " + " · ".join(ev[1]),
                                     font=ctk.CTkFont(size=10),
                                     text_color=TEXT_SEC,
                                     justify="left").grid(
                            row=1, column=0, padx=14, pady=(0, 8), sticky="w")
                    self._scroll_bottom()
                elif kind == "docs_changed":
                    self._refresh_doc_list()
                elif kind == "busy_done":
                    self.busy = False
                    self._set_inputs(True)
        except queue.Empty:
            pass
        self.after(50, self._drain)

    # ─────────────────────── worker glue ────────────────────────

    def _set_inputs(self, on: bool):
        state = "normal" if on else "disabled"
        self.btn_ask.configure(state=state)
        self.entry.configure(state=state)

    def _start(self, fn, *args):
        if self.busy:
            return
        self.busy = True
        self._set_inputs(False)
        def wrap():
            try:
                fn(*args)
            except Exception as e:
                self.events.put(("log", f"⚠ Error: {e}"))
                self.events.put(("status", "Error occurred."))
                self.events.put(("dot", DANGER))
            finally:
                self.events.put(("busy_done",))
        threading.Thread(target=wrap, daemon=True).start()

    # ─────────────────────── workers ────────────────────────────

    def _startup(self):
        try:
            listing = self.client.list()
        except Exception:
            self.events.put(("dot", DANGER))
            self.events.put(("status", "Ollama not running"))
            self.events.put(("log",
                "⚠ Cannot reach Ollama. Make sure the Ollama tray app is "
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
                f"{n} chunks · {len(self.store.files)} files"))
        else:
            self.events.put(("status", "Add documents to start"))
        self.events.put(("dot", SUCCESS))

    def _pull(self, model: str):
        self.events.put(("log",
            f"⬇ Downloading '{model}' (first run only)…"))
        last = -1
        for prog in self.client.pull(model, stream=True):
            total = rag._get(prog, "total") or 0
            done  = rag._get(prog, "completed") or 0
            if total:
                pct = int(done * 100 / total)
                if pct != last:
                    last = pct
                    self.events.put(("status",
                        f"Downloading {model}: {pct}%"))
        self.events.put(("log", f"✓ {model} ready"))

    def _ingest(self, paths):
        files = rag.collect_files(paths)
        if not files:
            self.events.put(("status", "No supported files found"))
            return
        for f in files:
            key = str(f.resolve())
            h   = rag.file_hash(f)
            if key in self.store.files and self.store.files[key]["hash"] == h:
                continue
            if key in self.store.files:
                for rid in self.store.files[key]["ids"]:
                    self.store.records.pop(rid, None)
            self.events.put(("status", f"Reading {f.name}…"))
            try:
                raw = rag.extract(f)
            except Exception as e:
                self.events.put(("log", f"⚠ Skipped {f.name}: {e}"))
                continue
            if not raw or not raw.strip():
                self.store.files[key] = {"hash": h, "ids": []}
                continue
            chunks = rag.chunk_text(raw, CFG["chunk_size"],
                                    CFG["chunk_overlap"])
            self.events.put(("status",
                f"Indexing {f.name} ({len(chunks)} chunks)…"))
            vecs = rag.embed_texts(self.client, CFG["embed_model"],
                                   chunks, CFG["keep_alive"])
            ids = []
            for ci, (ch, v) in enumerate(zip(chunks, vecs)):
                rid = self.store.next_id
                self.store.next_id += 1
                self.store.records[rid] = {
                    "text": ch, "source": str(f), "chunk": ci, "vec": v}
                ids.append(rid)
            self.store.files[key] = {"hash": h, "ids": ids}
            self.store.dim = int(vecs.shape[1])

        self.store.save()
        self.store.build_index()
        n = len(self.store.records)
        self.events.put(("status",
            f"{n} chunks · {len(self.store.files)} files"))
        self.events.put(("docs_changed",))

    def _ask(self, question: str):
        if not self.store.records:
            self.events.put(("log",
                "⚠ No documents indexed yet — add some first."))
            return
        self.store.build_index()
        qvec = rag.embed_texts(self.client, CFG["embed_model"],
                               [question], CFG["keep_alive"])
        hits = self.store.search(qvec, CFG["top_k"])
        messages = [
            {"role": "system",  "content": rag.SYSTEM_PROMPT},
            {"role": "user",    "content": rag.build_user_prompt(
                question, hits)},
        ]
        options = {"temperature": CFG["temperature"],
                   "num_ctx":     CFG["num_ctx"]}
        self.events.put(("bot_start",))
        for part in self.client.chat(
                model=CFG["llm_model"], messages=messages,
                options=options, keep_alive=CFG["keep_alive"],
                stream=True):
            token = rag._get(rag._get(part, "message"),
                             "content", "") or ""
            if token:
                self.events.put(("token", token))
        srcs = sorted({os.path.basename(r["source"])
                       for r, _ in hits}) if hits else []
        self.events.put(("bot_end", srcs))

    def _reset(self):
        (Path(CFG["store"]) / "meta.pkl").unlink(missing_ok=True)
        self.store = rag.VectorStore(
            CFG["store"], CFG["embed_model"]).load()
        self.events.put(("status", "Index cleared"))
        self.events.put(("docs_changed",))

    # ─────────────────────── callbacks ──────────────────────────

    def on_add_files(self):
        paths = filedialog.askopenfilenames(title="Choose documents")
        if paths:
            self._start(self._ingest, list(paths))

    def on_add_folder(self):
        folder = filedialog.askdirectory(title="Choose a folder")
        if folder:
            self._start(self._ingest, [folder])

    def on_reset(self):
        if messagebox.askyesno("Clear index",
                               "Remove all indexed documents?"):
            self._start(self._reset)

    def on_ask(self, _event=None):
        q = self.entry.get().strip()
        if not q or self.busy:
            return
        self.entry.delete(0, "end")
        self.events.put(("you", q))
        self._start(self._ask, q)


def main():
    app = RagApp()
    app.mainloop()


if __name__ == "__main__":
    main()