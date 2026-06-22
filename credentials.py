from __future__ import annotations

import hashlib
import json
import secrets

from theme import appdata_dir


class CredentialStore:
    def __init__(self):
        self.path = appdata_dir() / "users.json"

    def _load(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {}

    def _save(self, data: dict) -> None:
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def is_empty(self) -> bool:
        return not bool(self._load())

    def register(self, username: str, password: str) -> tuple[bool, str]:
        if not username.strip():
            return False, "Username cannot be empty."
        if len(password) < 6:
            return False, "Password must be at least 6 characters."
        data = self._load()
        if username.lower() in data:
            return False, "Username already taken."
        salt = secrets.token_hex(32)
        h = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), bytes.fromhex(salt), 200_000
        ).hex()
        data[username.lower()] = {"hash": h, "salt": salt, "display": username}
        self._save(data)
        return True, ""

    def authenticate(self, username: str, password: str) -> bool:
        data = self._load()
        entry = data.get(username.lower())
        if not entry:
            return False
        h = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), bytes.fromhex(entry["salt"]), 200_000
        ).hex()
        return h == entry["hash"]
