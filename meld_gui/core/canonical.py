"""Input canonicalization used by MELD v7+ checkpoints.

Ported verbatim from the reference ``meld.py`` that ships in the upstream
repository (MIT License, Copyright (c) the MELD authors; see NOTICE). Newer
checkpoints were trained and calibrated on canonical text: look-alike
characters folded, invisible characters removed and markdown stripped. Scoring
raw text against those thresholds would shift the false-positive rate.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

# ----------------------------------------------------------------- canonical form
_FENCE_LINE = re.compile(r"(?m)^[ \t]*```.*$")
_HEAD_MARK = re.compile(r"(?m)^[ \t]{0,3}#{1,6}[ \t]+")
_QUOTE_MARK = re.compile(r"(?m)^[ \t]{0,3}>[ \t]?")
_BULLET_MARK = re.compile(r"(?m)^[ \t]{0,3}(?:[-*+]|•|\U0001F539|✅|\U0001F4CC|➡️)[ \t]+")
_NUM_MARK = re.compile(r"(?m)^\d{1,3}[.)][ \t]+")
_TABLE_SEP = re.compile(r"(?m)^[ \t]{0,3}\|?[\s:|-]*-{2,}[\s:|-]*\|?[\s:|-]*$")
_EMPH = re.compile(r"(\*\*\*|\*\*|__)(?=\S)(.+?)(?<=\S)\1", re.S)
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_LINK = re.compile(r"\[([^\]\n]*)\]\([^)\n]*\)")
_WS = re.compile(r"[ \t]+")
_CELL_SPLIT = re.compile(r"(?<!\\)\|")
_ADDED = ("Overview", "Key points", "Section", "Point", "Detail", "Topic",
          "Let me know if you'd like more detail.")
_HOMOGLYPH_FOLD = {
    "а": "a", "А": "A", "Α": "A", "В": "B", "Β": "B",
    "е": "e", "Е": "E", "Ε": "E", "с": "c", "р": "p",
    "К": "K", "Κ": "K", "О": "O", "Ο": "O", "Р": "P",
    "Ρ": "P", "М": "M", "Μ": "M", "Н": "H", "Η": "H",
    "Т": "T", "Τ": "T", "Х": "X", "Χ": "X", "С": "C",
    "у": "y", "о": "o", "х": "x", "І": "I", "Ι": "I",
    "і": "i", "Ν": "N", "Ζ": "Z",
}
_INVISIBLE = dict.fromkeys((0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0x00AD, 0x180E), None)
_TYPO_FOLD = {
    0x2010: "-", 0x2011: "-", 0x2212: "-",
    0x00A0: " ", 0x202F: " ", 0x2007: " ", 0x2008: " ", 0x2009: " ", 0x200A: " ",
    0x2002: " ", 0x2003: " ", 0x3000: " ",
    0x2018: "'", 0x2019: "'", 0x201C: '"', 0x201D: '"', 0x2032: "'", 0x2033: '"',
}
_SURFACE_FOLD = {**_INVISIBLE, **_TYPO_FOLD}
_CONFUSABLE_CHARS = frozenset(_HOMOGLYPH_FOLD)
_TOKEN = re.compile(r"\S+")
# em dash runs: ", " inside a line, dropped at a line start, "," at a line end
_EM_DASH = re.compile("[ \t]*—+[ \t]*")
_EM_DASH_LEAD = re.compile("(?m)^[ \t]*—+[ \t]*")
_EM_DASH_TAIL = re.compile("(?m)[ \t]*—+[ \t]*$")


@lru_cache(maxsize=8192)
def _blocks_fold(ch: str) -> bool:
    if ch in _CONFUSABLE_CHARS or not ch.isalpha():
        return False
    try:
        return unicodedata.name(ch).startswith("CYRILLIC SMALL ")
    except ValueError:
        return False


def _fold_token(tok: str) -> str:
    if not any(c in _CONFUSABLE_CHARS for c in tok):
        return tok
    if any(_blocks_fold(c) for c in tok):
        return tok
    return "".join(_HOMOGLYPH_FOLD.get(c, c) for c in tok)


def fold_surface(text: str) -> str:
    """Delete invisible characters, fold em dashes to ", " and mixed-script look-alikes. Idempotent."""
    t = str(text or "").translate(_SURFACE_FOLD)
    if "—" in t:
        t = _EM_DASH.sub(", ", _EM_DASH_TAIL.sub(",", _EM_DASH_LEAD.sub("", t)))
    if not any(c in _CONFUSABLE_CHARS for c in t):
        return t
    return _TOKEN.sub(lambda m: _fold_token(m.group(0)), t)


def canonicalize(text: str) -> str:
    """Strip formatting, leaving a plain paragraph sequence. Idempotent."""
    t = fold_surface(text)
    t = _FENCE_LINE.sub("", t)
    t = _LINK.sub(r"\1", t)
    t = _INLINE_CODE.sub(r"\1", t)
    for _ in range(3):
        new = _EMPH.sub(lambda m: m.group(2), t)
        if new == t:
            break
        t = new
    out = []
    for line in t.split("\n"):
        if _TABLE_SEP.match(line):
            continue
        s = line.strip()
        if s.startswith("|") and len(_CELL_SPLIT.split(s)) >= 3:
            cells = [c.strip().replace(r"\|", "|") for c in _CELL_SPLIT.split(s.strip("|"))]
            cells = [c for c in cells if c and c not in _ADDED]
            line = " ".join(cells)
        for mark in (_HEAD_MARK, _QUOTE_MARK, _BULLET_MARK, _NUM_MARK):
            stripped = mark.sub("", line, count=1)
            if stripped != line:
                line = stripped
                break
        out.append(_WS.sub(" ", line).strip())
    kept = []
    for line in out:
        bare = re.sub(r"\s*\d+\s*$", "", line).strip().rstrip(":")
        if bare in _ADDED:
            continue
        kept.append(line)
    paras, cur = [], []
    for line in kept:
        if line:
            cur.append(line)
        elif cur:
            paras.append(" ".join(cur))
            cur = []
    if cur:
        paras.append(" ".join(cur))
    return "\n\n".join(p for p in paras if p.strip())

