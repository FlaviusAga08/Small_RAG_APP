#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

for _name in ("stdout", "stderr"):
    if getattr(sys, _name) is None:
        setattr(sys, _name, open(os.devnull, "w"))

from app import RagApp


def main():
    app = RagApp()
    app.mainloop()


if __name__ == "__main__":
    main()
