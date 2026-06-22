#!/usr/bin/env python3
"""
Quick connection test for the vast.ai LLM instance.
Run this before starting the app to verify everything is working.

Usage:
    python test_connection.py
"""
from __future__ import annotations

import os
import sys

try:
    from dotenv import load_dotenv
except ImportError:
    sys.exit("Run:  pip install python-dotenv")

try:
    from openai import OpenAI
except ImportError:
    sys.exit("Run:  pip install openai")

load_dotenv()

BASE_URL   = os.getenv("VAST_BASE_URL",   "http://localhost:11434/v1")
API_KEY    = os.getenv("VAST_API_KEY",    "not-required")
LLM_MODEL  = os.getenv("VAST_LLM_MODEL",  "llama3.2:3b")
EMBED_MODEL = os.getenv("VAST_EMBED_MODEL", "nomic-embed-text")

SEP   = "─" * 52
PASS  = "  ✓"
FAIL  = "  ✗"


def check(label: str, ok: bool, detail: str = "") -> bool:
    icon = PASS if ok else FAIL
    line = f"{icon}  {label}"
    if detail:
        line += f"  →  {detail}"
    print(line)
    return ok


def main():
    print(f"\n{SEP}")
    print("  vast.ai connection test")
    print(f"{SEP}")
    print(f"  URL:         {BASE_URL}")
    print(f"  LLM model:   {LLM_MODEL}")
    print(f"  Embed model: {EMBED_MODEL}")
    print(SEP)

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=30)

    # ── 1. Connectivity ─────────────────────────────────────────
    print("\n  1 / 3  — Connectivity")
    try:
        models = client.models.list()
        names  = [m.id for m in models.data]
        check("Server reachable", True, f"{len(names)} model(s) available")
        for n in names:
            print(f"          • {n}")
    except Exception as e:
        check("Server reachable", False, str(e))
        print(f"\n  {SEP}")
        print("  Make sure:")
        print("  • The vast.ai instance is running")
        print("  • VAST_BASE_URL in .env points to the correct IP and port")
        print("  • The port is open (check vast.ai dashboard → Open Ports)")
        print(f"  {SEP}\n")
        sys.exit(1)

    # ── 2. Chat completion ───────────────────────────────────────
    print("\n  2 / 3  — Chat completion")
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=10,
            temperature=0,
        )
        reply = resp.choices[0].message.content.strip()
        check("Chat works", True, f'model replied: "{reply}"')
    except Exception as e:
        check("Chat works", False, str(e))
        print(f"\n  Hint: make sure '{LLM_MODEL}' is pulled on the instance.")
        print(f"        SSH in and run:  ollama pull {LLM_MODEL}")

    # ── 3. Embeddings ────────────────────────────────────────────
    print("\n  3 / 3  — Embeddings")
    try:
        resp = client.embeddings.create(
            model=EMBED_MODEL,
            input=["test sentence"],
        )
        dim = len(resp.data[0].embedding)
        check("Embeddings work", True, f"vector dimension: {dim}")
    except Exception as e:
        check("Embeddings work", False, str(e))
        print(f"\n  Hint: make sure '{EMBED_MODEL}' is pulled on the instance.")
        print(f"        SSH in and run:  ollama pull {EMBED_MODEL}")

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Done. If all three checks passed, the app is ready to use.")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
