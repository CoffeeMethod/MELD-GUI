"""Splitting documents into spans that carry character offsets.

Everything downstream (the heatmap, the sentence table, the CSV export) needs
to point back at exact positions in the original string, so every splitter here
returns ``(start, end)`` offsets rather than substrings.  No NLP dependency is
pulled in for this: a tuned regex handles ordinary prose, and the cost of an
occasional mis-split is a slightly odd sentence boundary in the UI, not a wrong
score.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Abbreviations that end in a period without ending a sentence.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g",
    "i.e", "al", "fig", "no", "vol", "pp", "ed", "eds", "cf", "approx",
    "dept", "univ", "inc", "ltd", "co", "corp", "jan", "feb", "mar", "apr",
    "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}

_SENTENCE_END = re.compile(r"[.!?…]+[\"”’')\]]*\s+|\n{2,}")
_WORD = re.compile(r"\b[\w'’-]+\b", re.UNICODE)


@dataclass(slots=True)
class Span:
    """A character range in the source document."""

    start: int
    end: int
    text: str
    index: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }


def _looks_like_abbreviation(text: str, dot_pos: int) -> bool:
    """True when the period at ``dot_pos`` closes a known abbreviation."""
    head = text[max(0, dot_pos - 12) : dot_pos]
    token = re.split(r"[\s(\[\"]", head)[-1].lower().strip()
    if not token:
        return False
    if token in _ABBREVIATIONS:
        return True
    # Single initials such as "J." in "J. R. R. Tolkien".
    return len(token) == 1 and token.isalpha()


def split_sentences(text: str) -> list[Span]:
    """Split ``text`` into sentence spans, preserving offsets."""
    spans: list[Span] = []
    cursor = 0
    for match in _SENTENCE_END.finditer(text):
        end = match.end()
        boundary = match.start()
        if text[boundary : boundary + 1] == "." and _looks_like_abbreviation(text, boundary):
            continue
        chunk = text[cursor:end]
        if chunk.strip():
            spans.append(Span(cursor, end, chunk, len(spans)))
        cursor = end
    tail = text[cursor:]
    if tail.strip():
        spans.append(Span(cursor, len(text), tail, len(spans)))
    return spans


def split_paragraphs(text: str) -> list[Span]:
    """Split on blank lines, preserving offsets."""
    spans: list[Span] = []
    cursor = 0
    for match in re.finditer(r"\n\s*\n", text):
        chunk = text[cursor : match.start()]
        if chunk.strip():
            spans.append(Span(cursor, match.start(), chunk, len(spans)))
        cursor = match.end()
    tail = text[cursor:]
    if tail.strip():
        spans.append(Span(cursor, len(text), tail, len(spans)))
    return spans


def word_count(text: str) -> int:
    return len(_WORD.findall(text))


def text_stats(text: str) -> dict[str, float | int]:
    """Cheap readability-adjacent statistics shown beside the verdict.

    These are descriptive only. They are *not* part of the detection decision;
    the UI labels them as document statistics for exactly that reason.
    """
    words = _WORD.findall(text)
    sentences = [s for s in split_sentences(text) if s.text.strip()]
    n_words = len(words)
    n_sentences = len(sentences) or 1
    lengths = [len(w) for w in words] or [0]
    lowered = [w.lower() for w in words]
    unique = len(set(lowered))

    sent_words = [word_count(s.text) for s in sentences] or [0]
    mean_sent = sum(sent_words) / len(sent_words)
    variance = sum((x - mean_sent) ** 2 for x in sent_words) / len(sent_words)

    return {
        "characters": len(text),
        "words": n_words,
        "sentences": len(sentences),
        "paragraphs": len(split_paragraphs(text)),
        "unique_words": unique,
        "type_token_ratio": round(unique / n_words, 4) if n_words else 0.0,
        "avg_word_length": round(sum(lengths) / len(lengths), 2),
        "avg_sentence_words": round(mean_sent, 2),
        "sentence_length_sd": round(variance**0.5, 2),
    }
