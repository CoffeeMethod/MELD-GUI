# MELD GUI

A local web interface for [MELD](https://arxiv.org/abs/2605.06903), a
1B-parameter encoder that detects AI-generated English text.

Built for the current checkpoint, **MELD v83** (`MeldV7Detector`, Ettin-1B backbone).
Older v5 checkpoints still load and score with their own rules.

Everything runs on your machine. No text is sent anywhere.

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License](https://img.shields.io/badge/license-Apache%202.0-green)

<img width="2457" height="1652" alt="Screenshot 2026-09-06 230136" src="https://github.com/user-attachments/assets/64643c7e-c787-4dad-9e6e-a6f7dd8b5e86" />

<img width="2450" height="1643" alt="Screenshot 2026-09-06 230124" src="https://github.com/user-attachments/assets/d70c3abb-00af-4efc-94c1-492d007e6969" />

---

## Why this exists

The released checkpoint ships a short scorer (`meld.py`) that answers one question:
*what is `P(AI)` for this document?* That number alone is hard to act on. This app keeps
that computation identical and adds the context needed to actually read it:

- **Calibrated thresholds, not 0.5.** MELD's raw score is compared against the
  false-positive-rate thresholds shipped in the checkpoint, chosen per document type.
- **Where the evidence is.** The document score is the mean of the most machine-like 25%
  of tokens, so the per-token margins already exist. They are mapped back to character
  offsets and rendered as a sentence heatmap.
- **What it looks like.** The checkpoint carries 60 generator prototypes grouped into 7
  model families, plus 6 edit-operation prototypes. Both are surfaced (clearly labelled
  as indicative).
- **Long documents.** Longer than 2,048 tokens? Consecutive windows, pooled once across
  the whole text, up to the checkpoint's 16,384-token cap.

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate     # Windows
pip install -r requirements.txt
python -m meld_gui
```

Opens <http://127.0.0.1:8000>. On first run, click **Model → Download & load**
(~4.1 GB, cached in your HuggingFace folder afterwards).

### Updating the model

The app loads whatever revision is already in your cache. When the upstream repo
publishes a new checkpoint, click **Model → Download latest**: it asks Hugging Face for
the newest revision, downloads only the files that changed and loads it. Older
revisions stay in the cache untouched. The **Revision** row on the Model page shows the
commit you are running.

### Hugging Face access

The MELD repository is gated, so the first download needs a Hugging Face account
that has accepted the access conditions:

1. Open <https://huggingface.co/anon-review-meld-2026/meld> while signed in and accept the conditions.
2. Create a **read** token at <https://huggingface.co/settings/tokens>.
3. Either paste the token into **Model → Hugging Face token**, or set it before launch:

```bash
set MELD_HF_TOKEN=hf_xxx          # Windows cmd   (HF_TOKEN also works)
export MELD_HF_TOKEN=hf_xxx       # macOS / Linux
```

A token pasted into the UI is sent with that one request and kept nowhere. A token
from the environment is held in memory by the server process. Once the files are
cached, no token is needed (except for **Download latest**). `hf auth login` also works,
because the app falls back to whatever token `huggingface_hub` finds.

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

The v83 checkpoint was calibrated on 47,710 human documents across 16 document types and
ships the score below which 99% / 95% / 90% of them fell. Choosing **1% FPR** means about
one human document in a hundred *of that type* would be wrongly flagged. Those thresholds
vary a lot by genre, so pick the **Document type** that matches:

| Document type | 1% FPR | 5% FPR | 10% FPR | Calibration n |
|---|---:|---:|---:|---:|
| General / mixed | +3.92 | +2.30 | +0.66 | 47,710 |
| Business / finance | −2.51 | −4.87 | −5.84 | 3,000 |
| Consumer opinion | +2.31 | −0.52 | −1.94 | 3,000 |
| Correspondence | +3.92 | +2.30 | +0.66 | 2,999 |
| Creative writing | −0.19 | −2.26 | −3.42 | 3,000 |
| Dialogue / transcript | −1.96 | −4.66 | −5.63 | 3,000 |
| General web | +1.11 | −2.21 | −3.72 | 3,000 |
| Government / policy | −3.56 | −6.00 | −6.61 | 3,000 |
| Legal | −1.21 | −4.42 | −5.51 | 3,000 |
| News | +1.73 | −1.59 | −3.09 | 3,000 |
| Peer review | −2.26 | −5.00 | −6.01 | 2,981 |
| Poetry / lyrics | +1.85 | −2.38 | −3.93 | 2,887 |
| Reference / how-to | +0.36 | −2.48 | −3.56 | 2,999 |
| Scientific paper | −0.82 | −3.56 | −4.74 | 2,999 |
| Social / Q&A / chat | +3.83 | +0.18 | −1.65 | 2,974 |
| Student essay | +1.38 | −1.87 | −3.27 | 2,871 |
| Technical docs | −2.10 | −4.72 | −5.71 | 3,000 |

The app reads this table from whichever checkpoint is loaded, so the Document type list
always matches the model.

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
│   ├── canonical.py     Input cleanup from the reference meld.py (v7+ checkpoints)
│   ├── loader.py        Download / update / load lifecycle, threaded, UI progress
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

**The scoring maths is never re-derived.** `core/model.py` reproduces the reference
`meld.py` that ships with the checkpoint; a test asserts the analyzer matches it to 1e-4.
Everything else reads the intermediate tensors that computation already produces. If the
upstream checkpoint changes, that file and `core/canonical.py` are the things to update.

**Both checkpoint generations load.** v7+ stores one prototype per generator
(`gen_protos`) and maps each to a family through `generator_family`; a family's logit is
its nearest member's. v5 stored one prototype per family (`family_protos`), which loads
into the same layout with an identity map, exactly as the reference loader does.

**The scoring contract is read from the checkpoint.** v7+ `meld_config.json` carries a
`scoring_contract`. When it is present the analyzer follows it: the text is cleaned first
(look-alike characters folded, invisible characters and markdown removed, em dashes
rewritten), cut into consecutive 2,046-token windows with no overlap, capped at 16,384
tokens, and pooled once with top-ρ across all windows. Results then include
`scored_text`, the cleaned text that every offset refers to. Checkpoints without a
contract keep the older overlapping-window behaviour.

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
| `MELD_HF_TOKEN` | *(unset)* | Hugging Face read token for the gated repo (`HF_TOKEN` is also read). First download only |
| `MELD_DEVICE` | `auto` | `auto` / `cpu` / `cuda` |
| `MELD_BATCH_SIZE` | `4` | Windows per forward pass |
| `MELD_CHUNK_OVERLAP` | `128` | Token overlap between windows (only for checkpoints without a scoring contract; v83 uses none) |
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

Unit tests cover segmentation offsets, window planning, input cleanup, threshold logic,
extraction, model download and update, and every API endpoint. Integration tests (marked
`model`) load the cached checkpoint, assert parity with the reference scorer and check
that machine text outscores human text.

## Using the API directly

```python
import requests

requests.post("http://127.0.0.1:8000/api/model/load")   # once

r = requests.post("http://127.0.0.1:8000/api/analyze", json={
    "text": open("essay.txt").read(),
    "stratum": "student_essay",      # see /api/config for the list
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
- Only the first 16,384 tokens (roughly 12,000 words) of a document are scored; the app
  warns when it cuts a document short.
- Non-native English writing scores higher on most detectors; MELD's calibration set may
  not match your population.
- Formulaic prose (documentation, boilerplate, legal text) reads as machine-like.
- Heavily edited AI text lands in the inconclusive band — that band exists for a reason.

## Licence

**Apache License 2.0** — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

Two attributions matter:

- `meld_gui/core/model.py` and `meld_gui/core/canonical.py` port the reference scorer
  (`meld.py`) from the MELD repository, which is **MIT** licensed. MIT is compatible with Apache-2.0, and that notice is
  retained in [`NOTICE`](NOTICE) and in the file's own header.
- The **model weights are not redistributed here.** They are downloaded from the
  HuggingFace Hub at runtime and remain under their own MIT licence.

If you redistribute this project, Apache-2.0 §4 requires you to carry the `LICENSE` and
`NOTICE` files along with it.
