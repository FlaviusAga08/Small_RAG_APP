from __future__ import annotations

import customtkinter as ctk

from credentials import CredentialStore
from theme import (
    ACCENT, ACCENT_H, BG_MAIN, BG_SIDE, BG_INPUT, BG_CARD,
    BORDER, DANGER, TEXT_PRI, TEXT_SEC, TEXT_ACC,
)


class RegisterDialog(ctk.CTkToplevel):
    def __init__(self, parent, store: CredentialStore):
        super().__init__(parent)
        self.store = store
        self.registered = False
        self.registered_username = ""

        self.title("Create Account")
        self.resizable(False, False)
        self.grab_set()
        self.configure(fg_color=BG_MAIN)
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self._center(360, 390)

        ctk.CTkLabel(self, text="Create Account",
                     font=ctk.CTkFont(size=20, weight="bold"),
                     text_color=TEXT_ACC).pack(pady=(28, 16))

        frm = ctk.CTkFrame(self, fg_color=BG_SIDE, corner_radius=12)
        frm.pack(padx=30, fill="x")

        for label, attr, kw in [
            ("Username",         "user_entry", {}),
            ("Password",         "pass_entry", {"show": "●"}),
            ("Confirm password", "conf_entry", {"show": "●"}),
        ]:
            ctk.CTkLabel(frm, text=label, text_color=TEXT_SEC,
                         font=ctk.CTkFont(size=12)).pack(
                anchor="w", padx=16, pady=(12, 2))
            e = ctk.CTkEntry(frm, fg_color=BG_INPUT, border_color=BORDER,
                             text_color=TEXT_PRI, width=296, **kw)
            e.pack(padx=16)
            setattr(self, attr, e)

        self.conf_entry.bind("<Return>", lambda _: self._do_register())

        self.err_lbl = ctk.CTkLabel(self, text="", text_color=DANGER,
                                    font=ctk.CTkFont(size=11), wraplength=300)
        self.err_lbl.pack(pady=(10, 0))

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(pady=16)
        ctk.CTkButton(btn_row, text="Cancel", width=130,
                      fg_color=BG_CARD, hover_color=BORDER,
                      command=self.destroy).pack(side="left", padx=6)
        ctk.CTkButton(btn_row, text="Register", width=130,
                      fg_color=ACCENT, hover_color=ACCENT_H,
                      font=ctk.CTkFont(size=13, weight="bold"),
                      command=self._do_register).pack(side="left", padx=6)

        self.user_entry.focus()

    def _center(self, w: int, h: int) -> None:
        self.update_idletasks()
        sx, sy = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sx - w) // 2}+{(sy - h) // 2}")

    def _do_register(self):
        username = self.user_entry.get().strip()
        password = self.pass_entry.get()
        confirm  = self.conf_entry.get()
        if password != confirm:
            self.err_lbl.configure(text="Passwords do not match.")
            return
        ok, msg = self.store.register(username, password)
        if not ok:
            self.err_lbl.configure(text=msg)
            return
        self.registered = True
        self.registered_username = username
        self.destroy()
