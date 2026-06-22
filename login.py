from __future__ import annotations

import customtkinter as ctk

from credentials import CredentialStore
from register import RegisterDialog
from theme import (
    ACCENT, ACCENT_H, BG_MAIN, BG_SIDE, BG_INPUT,
    BORDER, DANGER, SUCCESS, WARNING,
    TEXT_PRI, TEXT_SEC, TEXT_ACC,
)


class LoginDialog(ctk.CTkToplevel):
    def __init__(self, parent, store: CredentialStore):
        super().__init__(parent)
        self.store = store
        self.authenticated = False
        self.user_info: dict | None = None

        self.title("Login")
        self.resizable(False, False)
        self.grab_set()
        self.configure(fg_color=BG_MAIN)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._center(360, 420)

        ctk.CTkLabel(self, text="Welcome back",
                     font=ctk.CTkFont(size=22, weight="bold"),
                     text_color=TEXT_ACC).pack(pady=(32, 4))
        ctk.CTkLabel(self, text="Sign in to continue",
                     font=ctk.CTkFont(size=12),
                     text_color=TEXT_SEC).pack()

        frm = ctk.CTkFrame(self, fg_color=BG_SIDE, corner_radius=12)
        frm.pack(padx=30, pady=20, fill="x")

        ctk.CTkLabel(frm, text="Username", text_color=TEXT_SEC,
                     font=ctk.CTkFont(size=12)).pack(
            anchor="w", padx=16, pady=(14, 2))
        self.user_entry = ctk.CTkEntry(frm, fg_color=BG_INPUT, border_color=BORDER,
                                       text_color=TEXT_PRI, width=296)
        self.user_entry.pack(padx=16)

        ctk.CTkLabel(frm, text="Password", text_color=TEXT_SEC,
                     font=ctk.CTkFont(size=12)).pack(
            anchor="w", padx=16, pady=(10, 2))
        self.pass_entry = ctk.CTkEntry(frm, show="●", fg_color=BG_INPUT,
                                       border_color=BORDER, text_color=TEXT_PRI,
                                       width=296)
        self.pass_entry.pack(padx=16, pady=(0, 14))
        self.pass_entry.bind("<Return>", lambda _: self._do_login())

        self.err_lbl = ctk.CTkLabel(self, text="", text_color=DANGER,
                                    font=ctk.CTkFont(size=11), wraplength=300)
        self.err_lbl.pack()

        ctk.CTkButton(self, text="Login", width=220, height=42,
                      fg_color=ACCENT, hover_color=ACCENT_H,
                      font=ctk.CTkFont(size=14, weight="bold"),
                      corner_radius=10,
                      command=self._do_login).pack(pady=(12, 8))

        sep = ctk.CTkFrame(self, fg_color="transparent")
        sep.pack()
        ctk.CTkLabel(sep, text="Don't have an account?",
                     font=ctk.CTkFont(size=11),
                     text_color=TEXT_SEC).pack(side="left")
        ctk.CTkButton(sep, text="Register", width=70,
                      fg_color="transparent", text_color=ACCENT,
                      hover_color=BG_SIDE,
                      font=ctk.CTkFont(size=11),
                      command=self._open_register).pack(side="left")

        if store.is_empty():
            self.err_lbl.configure(
                text="No accounts yet — click Register to get started.",
                text_color=WARNING)

        self.user_entry.focus()

    def _center(self, w: int, h: int) -> None:
        self.update_idletasks()
        sx, sy = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sx - w) // 2}+{(sy - h) // 2}")

    def _do_login(self):
        username = self.user_entry.get().strip()
        password = self.pass_entry.get()
        if self.store.authenticate(username, password):
            self.user_info = self.store.get_user(username)
            self.authenticated = True
            self.destroy()
        else:
            self.err_lbl.configure(text="Incorrect username or password.",
                                   text_color=DANGER)
            self.pass_entry.delete(0, "end")
            self.pass_entry.focus()

    def _open_register(self):
        dlg = RegisterDialog(self, self.store)
        self.wait_window(dlg)
        if dlg.registered:
            self.user_entry.delete(0, "end")
            self.user_entry.insert(0, dlg.registered_username)
            self.err_lbl.configure(
                text="Account created. Please log in.", text_color=SUCCESS)
            self.pass_entry.focus()

    def _on_close(self):
        self.authenticated = False
        self.destroy()
