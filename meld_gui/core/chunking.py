"""Token-window planning for documents longer than the model's context.

MELD reads at most ``max_length`` (2048) tokens. Longer documents are cut into
overlapping windows. The overlap exists so that no sentence sits permanently at
a window edge with only half its context; when two windows both cover a token,
the scorer keeps the copy that sat furthest from an edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Window:
    """One window of content tokens, in global token indices."""

    index: int
    start: int          # inclusive, global content-token index
    end: int            # exclusive
    ids: list[int] = field(default_factory=list)

    @property
    def length(self) -> int:
        return self.end - self.start

    def distance_from_edge(self, global_pos: int) -> int:
        """How central ``global_pos`` is inside this window."""
        return min(global_pos - self.start, self.end - 1 - global_pos)


def plan_windows(
    token_ids: list[int],
    max_length: int,
    overlap: int = 128,
) -> list[Window]:
    """Cut ``token_ids`` into windows sized for the model.

    ``max_length`` counts the special tokens the model adds, so the usable
    content width is two tokens smaller.
    """
    width = max(1, max_length - 2)
    n = len(token_ids)
    if n == 0:
        return []
    if n <= width:
        return [Window(0, 0, n, token_ids)]

    overlap = max(0, min(overlap, width - 1))
    stride = width - overlap

    windows: list[Window] = []
    start = 0
    while start < n:
        end = min(start + width, n)
        windows.append(Window(len(windows), start, end, token_ids[start:end]))
        if end >= n:
            break
        start += stride
    return windows
