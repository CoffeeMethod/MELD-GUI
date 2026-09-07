# MELD GUI

A local web interface for [MELD](https://huggingface.co/anon-review-meld-2026/meld), a
395M-parameter encoder that detects AI-generated English text.

Everything runs on your machine. No text is sent anywhere.

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License](https://img.shields.io/badge/license-Apache%202.0-green)

---

## Why this exists

The released checkpoint ships a ~30-line scorer that answers one question: *what is
`P(AI)` for this document?* That number alone is hard to act on. This app keeps that
computation bit-for-bit identical and adds the context needed to actually read it:

- **Calibrated thresholds, not 0.5.** MELD's raw score is compared against the
  false-positive-rate thresholds shipped in the checkpoint, chosen per document type.
- **Where the evidence is.** The document score is the mean of the most machine-like 25%
  of tokens, so the per-token margins already exist. They are mapped back to character
  offsets and rendered as a sentence heatmap.
- **What it looks like.** The checkpoint carries 11 model-family prototypes and 9
  edit-operation prototypes. Both are surfaced (clearly labelled as indicative).
- **Documents of any length.** Longer than 2,048 tokens? Overlapping windows, pooled.

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate     # Windows
pip install -r requirements.txt
python -m meld_gui
```

Opens <http://127.0.0.1:8000>. On first run, click **Model → Download & load**
(~1.6 GB, cached in your HuggingFace folder afterwards).

```bash
python -m meld_gui --preload            # download and load at startup
python -m meld_gui --device cpu         # force CPU
python -m meld_gui --port 9000 --no-browser
python -m meld_gui --model-dir ./meld   # use a local checkpoint folder
```

## Features

| | |
|---|---|
| **Analyze** | Paste or drop a file. Gauge, calibrated score scale, sentence heatmap, sortable sentence table, attribution, per-window breakdown, statistics, raw JSON. |
| **Batch** | Drop many files; results stream in as each finishes. CSV export. |
| **Compare** | Two documents scored under identical settings, with the raw-score delta. |
| **History** | Every run stored in local SQLite, searchable, re-openable, CSV export. |
| **Model** | Download progress, device selection, live checkpoint and calibration inspection. |
| **REST API** | The UI is a client of the app's own API. Docs at `/docs`. |

Files supported: `.txt` `.md` `.rst` `.csv` `.json` `.tex` `.log` · `.pdf` · `.docx`

## Reading a result

MELD's decision happens in **raw-score** space. The probability on the gauge is just
`sigmoid(raw score)` — a squashed view of the same number, not a calibrated likelihood.

The checkpoint was calibrated on 12,461 human documents and ships the score below which
99% / 95% / 90% of them fell. Choosing **1% FPR** means about one human document in a
hundred *of that type* would be wrongly flagged. Those thresholds vary a lot by genre:

| Document type | 1% FPR | 5% FPR | 10% FPR |
|---|---:|---:|---:|
| General / mixed | +1.92 | −0.47 | −1.26 |
| Academic | +1.21 | −1.15 | −1.44 |
| Creative | +2.39 | +0.44 | −0.61 |
| Q&A / social | +2.62 | +1.35 | +0.19 |
| Reviews | +2.80 | +0.80 | −0.13 |
| Web prose | +1.43 | −1.03 | −1.43 |
| Encyclopedic | +1.91 | −0.63 | −1.35 |

> **A flag is not proof.** At 1% FPR, scanning 500 human essays still yields ~5 false
> accusations. Use this to prioritise human review, never as the sole basis for a
> decision about a person. Below 100 words the score is noise, and the app says so.

## Architecture

```
meld_gui/
├── config.py            Settings; every value overridable via MELD_* env vars
├── app.py               FastAPI factory
├── __main__.py          CLI entry point
├── core/
│   ├── model.py         Faithful port of the released scoring head, extended to
│   │                    return per-token / family / operator tensors
│   ├── loader.py        Download + load lifecycle, threaded, progress for the UI
│   ├── analyzer.py      Windowing, stitching, sentence aggregation, attribution
│   ├── chunking.py      Token-window planning for long documents
│   ├── segmentation.py  Sentence/paragraph splitting with character offsets
│   ├── thresholds.py    Calibration table and verdict banding
│   └── extract.py       Text extraction from txt/md/pdf/docx
├── api/
│   ├── routes.py        Every endpoint the frontend uses
│   └── schemas.py       Pydantic request models
├── services/
│   ├── history.py       SQLite store (one table, no ORM)
│   └── jobs.py          In-memory batch job registry
└── web/                 Zero-build frontend: no npm, no bundler
    ├── templates/index.html
    └── static/{css/app.css, js/{api,utils,charts,app}.js}
```

### Design decisions worth knowing

**The scoring maths is never re-derived.** `core/model.py` reproduces the model card's
`score()` exactly; a test asserts parity to 1e-4. Everything else reads the intermediate
tensors that computation already produces. If the upstream checkpoint changes, that one
file is the thing to update.

**Long documents.** `plan_windows()` cuts the token stream into overlapping windows.
Where two windows cover the same token, the copy that sat furthest from a window edge
wins — it was scored with the most context. The document score then applies the released
top-ρ pooling across the whole document, not per window.

**Attribution is labelled indicative.** The family and operator prototypes are in the
checkpoint but the model card publishes no accuracy figures for them, so the UI says so
rather than presenting them as identification.

**No frontend build step.** Four plain `.js` files and one `.css`. Deliberate: it keeps
the app runnable years from now without a dead toolchain.

## Extending it

| To add… | Touch |
|---|---|
| A new analysis output | `core/analyzer.py` → add to the result dict, render in `web/static/js/app.js` |
| A new file format | `core/extract.py` → add a suffix set and a branch |
| A new endpoint | `api/routes.py` (auto-appears in `/docs`) |
| Different thresholds/bands | `core/thresholds.py` → `_BANDS` |
| Restyling | `web/static/css/app.css` → the two `:root` blocks hold every colour |
| A new view | Add a `<section class="view">` in `index.html` plus a `.nav-item` |

### Configuration

All settings take a `MELD_` prefixed environment variable:

| Variable | Default | Meaning |
|---|---|---|
| `MELD_REPO_ID` | `anon-review-meld-2026/meld` | Checkpoint to fetch |
| `MELD_MODEL_DIR` | *(unset)* | Use a local folder instead of the HF cache |
| `MELD_DEVICE` | `auto` | `auto` / `cpu` / `cuda` |
| `MELD_BATCH_SIZE` | `4` | Windows per forward pass |
| `MELD_CHUNK_OVERLAP` | `128` | Token overlap between windows |
| `MELD_MAX_CHARS` | `400000` | Largest accepted document |
| `MELD_MIN_WORDS` | `100` | Reliability floor for the warning |
| `MELD_HISTORY_LIMIT` | `500` | Rows kept before trimming |
| `MELD_EAGER_LOAD` | `0` | Load the checkpoint at startup |

## Testing

```bash
pip install pytest httpx
pytest                    # everything
pytest -m "not model"     # logic only — no checkpoint needed
```

53 tests: segmentation offsets, window planning, threshold logic, extraction, every API
endpoint, plus integration tests that assert reference parity with the model card and
that machine text outscores human text.

## Using the API directly

```python
import requests

requests.post("http://127.0.0.1:8000/api/model/load")   # once

r = requests.post("http://127.0.0.1:8000/api/analyze", json={
    "text": open("essay.txt").read(),
    "stratum": "academic",
    "fpr": "fpr_0.01",
}).json()

d = r["decision"]
print(f"{d['probability']:.1%}  raw={d['score']:+.2f}  flagged={d['flagged']}")
for s in sorted(r["sentences"], key=lambda s: -s["score"])[:3]:
    print(f"  {s['score']:+.2f}  {s['text'][:70]}")
```

## Limitations

- English only.
- Below 100 words, results are unreliable regardless of author.
- Non-native English writing scores higher on most detectors; MELD's calibration set may
  not match your population.
- Formulaic prose (documentation, boilerplate, legal text) reads as machine-like.
- Heavily edited AI text lands in the inconclusive band — that band exists for a reason.

## Licence

**Apache License 2.0** — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

Two attributions matter:

- `meld_gui/core/model.py` ports the reference scoring head from the MELD model card,
  which is **MIT** licensed. MIT is compatible with Apache-2.0, and that notice is
  retained in [`NOTICE`](NOTICE) and in the file's own header.
- The **model weights are not redistributed here.** They are downloaded from the
  HuggingFace Hub at runtime and remain under their own MIT licence.

If you redistribute this project, Apache-2.0 §4 requires you to carry the `LICENSE` and
`NOTICE` files along with it.
