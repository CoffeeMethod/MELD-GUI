"""API and end-to-end tests.

Tests that need the 1.6 GB checkpoint are marked ``model`` and skip themselves
when it is not present, so the suite stays runnable on a clean checkout:

    pytest                    # everything available
    pytest -m "not model"     # logic only, no download
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from meld_gui.app import create_app
from meld_gui.core.loader import manager

HUMAN = (
    "I never really planned to keep chickens. My neighbour Dave turned up one Saturday "
    "with a cardboard box, said his brother was moving to a flat in Leeds and could not "
    "take them, and that was that. Three hens, one of them missing most of her tail "
    "feathers. I built the coop badly, out of pallet wood and a lot of swearing, and the "
    "door still sticks when it rains. But they lay well enough, and the small one, the "
    "one my daughter named Biscuit, follows me around the garden like a dog. My wife "
    "thinks this is ridiculous. She is probably right about that, as she is about most "
    "things involving the garden, but I have stopped arguing the point with her."
)

MACHINE = (
    "Artificial intelligence has fundamentally transformed the landscape of modern "
    "industry. By leveraging advanced machine learning algorithms, organizations can "
    "unlock unprecedented insights from their data assets. Furthermore, the integration "
    "of AI-driven solutions enables businesses to streamline operations, enhance "
    "decision-making processes, and deliver superior customer experiences. It is "
    "important to note that successful implementation requires careful consideration of "
    "several key factors. First, organizations must establish a robust data "
    "infrastructure. Second, they must cultivate a culture of innovation and continuous "
    "learning. Finally, ethical considerations must remain at the forefront of any AI "
    "initiative, ensuring that these powerful technologies are deployed responsibly."
)

needs_model = pytest.mark.model


@pytest.fixture(scope="session")
def client():
    with TestClient(create_app()) as c:
        yield c


@pytest.fixture(scope="session")
def loaded_model():
    """Load the checkpoint once, or skip every test that needs it."""
    if manager.local_path() is None:
        pytest.skip("MELD checkpoint not downloaded")
    if not manager.is_loaded:
        manager.load_blocking()
    return manager


# --------------------------------------------------------------------- no model
class TestMeta:
    def test_health(self, client):
        body = client.get("/api/health").json()
        assert body["status"] == "ok"
        assert "version" in body

    def test_config_lists_strata_and_fpr_levels(self, client):
        body = client.get("/api/config").json()
        keys = {s["key"] for s in body["strata"]}
        assert {"overall", "academic", "web", "wiki", "creative", "reviews"} <= keys
        assert [f["key"] for f in body["fpr_levels"]] == ["fpr_0.01", "fpr_0.05", "fpr_0.1"]
        assert body["min_words"] == 100

    def test_index_page_serves(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "MELD" in response.text

    def test_model_status_shape(self, client):
        body = client.get("/api/model").json()
        assert {"loaded", "busy", "downloaded", "device"} <= set(body)

    def test_empty_text_is_rejected(self, client):
        assert client.post("/api/analyze", json={"text": "   "}).status_code == 400

    def test_extract_endpoint(self, client):
        files = {"file": ("doc.txt", io.BytesIO(b"Hello there, world."), "text/plain")}
        body = client.post("/api/extract", files=files).json()
        assert body["text"] == "Hello there, world."
        assert body["statistics"]["words"] == 3

    def test_extract_rejects_unknown_type(self, client):
        files = {"file": ("x.png", io.BytesIO(b"\x89PNG"), "image/png")}
        assert client.post("/api/extract", files=files).status_code == 400

    def test_compare_needs_two_documents(self, client):
        response = client.post("/api/compare", json={"documents": [{"text": "one"}]})
        assert response.status_code == 400


# ------------------------------------------------------------------ with model
@needs_model
class TestAnalysis:
    def test_reference_parity(self, loaded_model):
        """Our analyzer must reproduce the model card's scorer exactly."""
        from meld_gui.core.analyzer import AnalysisOptions, analyze_text

        model, tokenizer, _, device = loaded_model.require()
        _, reference = model.score([MACHINE], tokenizer, device)
        ours = analyze_text(MACHINE, AnalysisOptions())["decision"]["score"]
        assert ours == pytest.approx(reference[0], abs=1e-4)

    def test_machine_scores_above_human(self, loaded_model):
        from meld_gui.core.analyzer import AnalysisOptions, analyze_text

        human = analyze_text(HUMAN, AnalysisOptions())
        machine = analyze_text(MACHINE, AnalysisOptions())
        assert machine["decision"]["score"] > human["decision"]["score"]
        assert machine["decision"]["flagged"] is True
        assert human["decision"]["flagged"] is False

    def test_result_contains_every_section(self, client, loaded_model):
        body = client.post("/api/analyze", json={"text": MACHINE, "save_history": False}).json()
        for key in ("decision", "statistics", "tokens", "timing",
                    "attribution", "sentences", "chunks", "token_detail"):
            assert key in body, key

    def test_sentence_offsets_map_back_to_the_source(self, client, loaded_model):
        body = client.post("/api/analyze", json={"text": MACHINE, "save_history": False}).json()
        for row in body["sentences"]:
            assert MACHINE[row["start"] : row["end"]].strip() == row["text"]

    def test_token_offsets_are_ordered_and_in_range(self, client, loaded_model):
        body = client.post("/api/analyze", json={"text": MACHINE, "save_history": False}).json()
        offsets = body["token_detail"]["offsets"]
        assert len(offsets) == len(body["token_detail"]["scores"])
        assert all(0 <= a <= b <= len(MACHINE) for a, b in offsets)
        assert offsets == sorted(offsets)

    def test_attribution_probabilities_sum_to_one(self, client, loaded_model):
        body = client.post("/api/analyze", json={"text": MACHINE, "save_history": False}).json()
        for key in ("families", "operations"):
            total = sum(r["probability"] for r in body["attribution"][key])
            assert total == pytest.approx(1.0, abs=1e-3)

    def test_stratum_changes_the_threshold(self, client, loaded_model):
        base = {"text": MACHINE, "save_history": False}
        overall = client.post("/api/analyze", json={**base, "stratum": "overall"}).json()
        academic = client.post("/api/analyze", json={**base, "stratum": "academic"}).json()
        assert overall["decision"]["score"] == pytest.approx(academic["decision"]["score"])
        assert overall["decision"]["threshold"] != academic["decision"]["threshold"]

    def test_looser_fpr_lowers_the_bar(self, client, loaded_model):
        base = {"text": MACHINE, "save_history": False}
        strict = client.post("/api/analyze", json={**base, "fpr": "fpr_0.01"}).json()
        loose = client.post("/api/analyze", json={**base, "fpr": "fpr_0.1"}).json()
        assert loose["decision"]["threshold"] < strict["decision"]["threshold"]

    def test_short_text_is_warned_about(self, client, loaded_model):
        body = client.post(
            "/api/analyze", json={"text": "Too short to judge.", "save_history": False}
        ).json()
        assert any(w["level"] == "warning" for w in body["warnings"])

    def test_long_document_uses_multiple_windows(self, client, loaded_model):
        long_text = " ".join([MACHINE] * 25)
        body = client.post(
            "/api/analyze", json={"text": long_text, "save_history": False}
        ).json()
        assert body["tokens"]["windows"] > 1
        assert len(body["chunks"]) == body["tokens"]["windows"]

    def test_file_upload_round_trip(self, client, loaded_model):
        files = {"file": ("essay.txt", io.BytesIO(MACHINE.encode()), "text/plain")}
        response = client.post("/api/analyze/file", files=files, data={"stratum": "academic"})
        assert response.status_code == 200
        assert response.json()["source_file"]["name"] == "essay.txt"

    def test_compare_returns_both_documents(self, client, loaded_model):
        body = client.post("/api/compare", json={
            "documents": [{"name": "H", "text": HUMAN}, {"name": "M", "text": MACHINE}]
        }).json()
        assert len(body["documents"]) == 2
        scores = [d["result"]["decision"]["score"] for d in body["documents"]]
        assert scores[1] > scores[0]


@needs_model
class TestHistoryAndBatch:
    def test_history_round_trip(self, client, loaded_model):
        created = client.post(
            "/api/analyze", json={"text": MACHINE, "label": "unit-test", "save_history": True}
        ).json()
        record_id = created["record_id"]

        listing = client.get("/api/history", params={"q": "unit-test"}).json()
        assert any(i["id"] == record_id for i in listing["items"])

        record = client.get(f"/api/history/{record_id}").json()
        assert record["label"] == "unit-test"
        assert record["payload"]["decision"]["score"] == created["decision"]["score"]

        assert client.delete(f"/api/history/{record_id}").json()["deleted"] is True
        assert client.get(f"/api/history/{record_id}").status_code == 404

    def test_history_csv_export(self, client, loaded_model):
        response = client.get("/api/history/export.csv")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/csv")

    def test_batch_runs_to_completion(self, client, loaded_model):
        import time

        job = client.post("/api/batch", json={
            "documents": [
                {"name": "human.txt", "text": HUMAN},
                {"name": "machine.txt", "text": MACHINE},
            ],
            "save_history": False,
        }).json()

        deadline = time.time() + 90
        while time.time() < deadline:
            status = client.get(f"/api/batch/{job['id']}").json()
            if status["status"] in {"done", "error", "cancelled"}:
                break
            time.sleep(0.3)

        assert status["status"] == "done"
        assert status["completed"] == 2
        by_name = {i["name"]: i for i in status["items"]}
        assert by_name["machine.txt"]["probability"] > by_name["human.txt"]["probability"]

    def test_batch_rejects_empty_input(self, client, loaded_model):
        assert client.post("/api/batch", json={"documents": []}).status_code == 400

    def test_unknown_batch_job(self, client):
        assert client.get("/api/batch/does-not-exist").status_code == 404
