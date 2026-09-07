"""Unit tests for the pure-logic layers. These run without the checkpoint."""

from __future__ import annotations

import json

import pytest

from meld_gui.core.chunking import plan_windows
from meld_gui.core.extract import ExtractionError, extract
from meld_gui.core.segmentation import (
    split_paragraphs,
    split_sentences,
    text_stats,
    word_count,
)
from meld_gui.core.thresholds import DEFAULT_FPR, OVERALL, ThresholdTable

# A trimmed copy of the shape shipped in meld_config.json.
OFFSETS = {
    "overall": {"fpr_0.01": 1.9156, "fpr_0.05": -0.4680, "fpr_0.1": -1.2608},
    "n_human": 12461,
    "strata": {
        "academic": {"n": 1654, "fpr_0.01": 1.2058, "fpr_0.05": -1.1495, "fpr_0.1": -1.4352},
        "web": {"n": 8028, "fpr_0.01": 1.4255, "fpr_0.05": -1.0260, "fpr_0.1": -1.4307},
    },
}


# ----------------------------------------------------------------- segmentation
class TestSegmentation:
    def test_sentences_cover_the_whole_document(self):
        text = "First one. Second one! Third one? Fourth."
        spans = split_sentences(text)
        assert len(spans) == 4
        # Offsets must reconstruct the original string exactly.
        assert "".join(text[s.start : s.end] for s in spans) == text

    def test_abbreviations_do_not_split(self):
        text = "Dr. Smith met Mr. Jones at 4pm. They talked for an hour."
        assert len(split_sentences(text)) == 2

    def test_initials_do_not_split(self):
        assert len(split_sentences("J. R. R. Tolkien wrote books. He liked trees.")) == 2

    def test_blank_line_separates_sentences(self):
        assert len(split_sentences("No terminal punctuation\n\nSecond block")) == 2

    def test_paragraphs(self):
        assert len(split_paragraphs("One.\n\nTwo.\n\n\nThree.")) == 3

    def test_empty_document(self):
        assert split_sentences("") == []
        assert split_sentences("   \n  ") == []

    def test_word_count_handles_punctuation_and_apostrophes(self):
        assert word_count("It's a well-known fact, isn't it?") == 6

    def test_stats_shape(self):
        stats = text_stats("Alpha beta gamma. Delta epsilon.\n\nZeta eta theta iota.")
        assert stats["words"] == 9
        assert stats["sentences"] == 3
        assert stats["paragraphs"] == 2
        assert 0 < stats["type_token_ratio"] <= 1


# --------------------------------------------------------------------- chunking
class TestChunking:
    def test_short_document_is_one_window(self):
        windows = plan_windows(list(range(100)), max_length=2048)
        assert len(windows) == 1
        assert (windows[0].start, windows[0].end) == (0, 100)

    def test_long_document_is_split_with_overlap(self):
        ids = list(range(5000))
        windows = plan_windows(ids, max_length=2048, overlap=128)
        assert len(windows) > 1
        # Windows must cover every token.
        assert windows[0].start == 0
        assert windows[-1].end == len(ids)
        # Consecutive windows must genuinely overlap.
        for a, b in zip(windows, windows[1:]):
            assert b.start < a.end

    def test_window_width_respects_special_tokens(self):
        windows = plan_windows(list(range(5000)), max_length=2048, overlap=0)
        assert all(w.length <= 2046 for w in windows)

    def test_empty_input(self):
        assert plan_windows([], max_length=2048) == []

    def test_overlap_larger_than_width_is_clamped(self):
        windows = plan_windows(list(range(500)), max_length=10, overlap=999)
        assert len(windows) > 1
        assert windows[-1].end == 500

    def test_distance_from_edge(self):
        window = plan_windows(list(range(10)), max_length=12)[0]
        assert window.distance_from_edge(0) == 0
        assert window.distance_from_edge(5) == 4


# ------------------------------------------------------------------ thresholds
class TestThresholds:
    @pytest.fixture()
    def table(self):
        return ThresholdTable(json.loads(json.dumps(OFFSETS)))

    def test_strata_listing(self, table):
        assert table.strata == ["overall", "academic", "web"]
        assert table.n_human() == 12461

    def test_threshold_lookup(self, table):
        assert table.threshold("overall", "fpr_0.01") == pytest.approx(1.9156)
        assert table.threshold("academic", "fpr_0.1") == pytest.approx(-1.4352)

    def test_unknown_keys_raise(self, table):
        with pytest.raises(KeyError):
            table.threshold("nonsense")
        with pytest.raises(KeyError):
            table.threshold("overall", "fpr_0.5")

    def test_flag_is_strictly_above_threshold(self, table):
        threshold = table.threshold(OVERALL, DEFAULT_FPR)
        assert table.decide(threshold + 0.01, 0.9).flagged is True
        assert table.decide(threshold, 0.9).flagged is False

    def test_verdict_bands_are_ordered(self, table):
        threshold = table.threshold()
        verdicts = [
            table.decide(threshold + 3.0, 0.99).verdict,
            table.decide(threshold + 0.5, 0.9).verdict,
            table.decide(threshold - 1.0, 0.5).verdict,
            table.decide(threshold - 5.0, 0.1).verdict,
        ]
        assert verdicts == ["ai", "likely_ai", "uncertain", "human"]

    def test_stratum_changes_the_decision(self, table):
        """The same score can flag under one stratum and not another."""
        score = 1.6
        assert table.decide(score, 0.8, "academic").flagged is True
        assert table.decide(score, 0.8, "overall").flagged is False

    def test_describe_is_ui_ready(self, table):
        rows = table.describe()
        assert {r["key"] for r in rows} == {"overall", "academic", "web"}
        assert all("label" in r and "thresholds" in r for r in rows)

    def test_fpr_percentage(self, table):
        assert table.decide(0.0, 0.5, fpr="fpr_0.05").fpr_pct == pytest.approx(5.0)


# --------------------------------------------------------------------- extract
class TestExtract:
    def test_plain_text(self):
        result = extract(b"Hello world", "note.txt")
        assert result.text == "Hello world"
        assert result.kind == "text"

    def test_utf8_bom_and_fallback_encodings(self):
        assert extract("café".encode("utf-8-sig"), "a.txt").text.endswith("café")
        assert extract("café".encode("cp1252"), "b.txt").text == "café"

    def test_markdown_is_treated_as_text(self):
        assert extract(b"# Title\n\nBody", "readme.md").kind == "text"

    def test_empty_file_rejected(self):
        with pytest.raises(ExtractionError):
            extract(b"   ", "empty.txt")

    def test_unsupported_extension_rejected(self):
        with pytest.raises(ExtractionError, match="Unsupported"):
            extract(b"\x00\x01", "image.png")

    def test_corrupt_pdf_reports_cleanly(self):
        with pytest.raises(ExtractionError):
            extract(b"not really a pdf", "broken.pdf")
