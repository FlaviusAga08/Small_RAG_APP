#!/usr/bin/env python3
"""
Admin CLI — manage user subscriptions.

Usage:
  python admin.py list
  python admin.py trial    <username> [days=30]
  python admin.py activate <username>
  python admin.py revoke   <username>
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from datetime import date
from credentials import CredentialStore


def cmd_list():
    store = CredentialStore()
    data  = store._load()
    if not data:
        print("  No users registered yet.")
        return

    print(f"\n  {'USERNAME':<22} {'STATUS':<20} {'TRIAL EXPIRES':<14} DISPLAY")
    print("  " + "─" * 70)
    for key, entry in sorted(data.items()):
        raw_status   = entry.get("status", "active")
        trial_str    = entry.get("trial_until", "")
        display      = entry.get("display", key)

        if raw_status == "trial" and trial_str:
            days = (date.fromisoformat(trial_str) - date.today()).days
            if days > 0:
                status_col = f"trial ({days}d left)"
            else:
                status_col = "trial (EXPIRED)"
        elif raw_status == "active":
            status_col = "active"
            trial_str  = "—"
        else:
            status_col = "revoked"
            trial_str  = "—"

        print(f"  {key:<22} {status_col:<20} {trial_str or '—':<14} {display}")
    print()


def cmd_trial(username: str, days: int = 30):
    store = CredentialStore()
    if store.set_status(username, "trial", trial_days=days):
        print(f"  ✓  {username}  →  trial for {days} days")
    else:
        print(f"  ✗  User '{username}' not found.")


def cmd_activate(username: str):
    store = CredentialStore()
    if store.set_status(username, "active"):
        print(f"  ✓  {username}  →  active (paid)")
    else:
        print(f"  ✗  User '{username}' not found.")


def cmd_revoke(username: str):
    store = CredentialStore()
    if store.set_status(username, "revoked"):
        print(f"  ✓  {username}  →  revoked")
    else:
        print(f"  ✗  User '{username}' not found.")


USAGE = """
Usage:
  python admin.py list
  python admin.py trial    <username> [days]    default: 30 days
  python admin.py activate <username>
  python admin.py revoke   <username>
"""


def main():
    args = sys.argv[1:]
    if not args:
        print(USAGE)
        return

    cmd = args[0].lower()

    if cmd == "list":
        cmd_list()

    elif cmd == "trial":
        if len(args) < 2:
            print("  Usage: python admin.py trial <username> [days]")
            return
        days = int(args[2]) if len(args) > 2 else 30
        cmd_trial(args[1], days)

    elif cmd == "activate":
        if len(args) < 2:
            print("  Usage: python admin.py activate <username>")
            return
        cmd_activate(args[1])

    elif cmd == "revoke":
        if len(args) < 2:
            print("  Usage: python admin.py revoke <username>")
            return
        cmd_revoke(args[1])

    else:
        print(f"  Unknown command: {cmd}")
        print(USAGE)


if __name__ == "__main__":
    main()
