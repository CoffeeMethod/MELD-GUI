#!/usr/bin/env python
"""Convenience launcher: ``python run.py``.

Equivalent to ``python -m meld_gui``; kept so the project can be started from
an IDE run configuration without arguments.
"""

import sys

from meld_gui.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
