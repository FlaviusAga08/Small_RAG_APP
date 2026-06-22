from __future__ import annotations

import os
from pathlib import Path

import customtkinter as ctk

APP_NAME  = "OfflineRAG"
APP_TITLE = "RAG"
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
