from __future__ import annotations

import hashlib
import json
import secrets
from datetime import date, timedelta

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
        trial_until = (date.today() + timedelta(days=30)).isoformat()
        data[username.lower()] = {
            "hash": h, "salt": salt, "display": username,
            "status": "trial", "trial_until": trial_until,
        }
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

    def get_user(self, username: str) -> dict | None:
        """Return user info with computed effective subscription status."""
        data = self._load()
        entry = data.get(username.lower())
        if not entry:
            return None

        stored_status = entry.get("status", "active")   # legacy accounts → active
        trial_until_str = entry.get("trial_until")
        days_remaining = None

        if stored_status == "trial" and trial_until_str:
            days_remaining = (date.fromisoformat(trial_until_str) - date.today()).days
            effective_status = "trial" if days_remaining > 0 else "trial_expired"
            days_remaining = max(days_remaining, 0)
        elif stored_status == "active":
            effective_status = "active"
        else:
            effective_status = "revoked"

        return {
            "username":      username.lower(),
            "display":       entry.get("display", username),
            "status":        effective_status,
            "days_remaining": days_remaining,
        }

    def set_status(self, username: str, status: str,
                   trial_days: int = 30) -> bool:
        """Admin: change a user's subscription status. Returns False if not found."""
        data = self._load()
        key = username.lower()
        if key not in data:
            return False
        data[key]["status"] = status
        if status == "trial":
            data[key]["trial_until"] = (
                date.today() + timedelta(days=trial_days)
            ).isoformat()
        else:
            data[key].pop("trial_until", None)
        self._save(data)
        return True
