# SyllabusLens

An academic workspace that organises course material **by subject**, then shows how question papers relate to lecture notes: which sections each question draws on, what the papers never touch, and what to revise.

Matching is **lexical** (TF-IDF / Bag-of-Words cosine similarity). It measures shared vocabulary, not understanding. It does not predict future exam questions.

> **Status: local, single-user application.** It has no sign-in and is not ready for multi-user or public deployment. See [Before a real deployment](#before-a-real-deployment).

## Quick start

Python 3.12 is the tested version.

```bash
# Linux / macOS
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

```powershell
# Windows (PowerShell)
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open the **Diagnostics** page first on a new machine. It checks every dependency and extraction path and reports exactly what fails.

Run the tests:

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

Optional environment variables: `SYLLABUSLENS_DATA_DIR` (where the database and uploads live; default `./data`) and `SYLLABUSLENS_MAX_UPLOAD_MB` (default 25; keep `maxUploadSize` in `.streamlit/config.toml` in step).

## What you can do

| Page | Purpose |
|---|---|
| Overview | Subject status, the next recommended action, latest findings, recent analyses |
| Subjects | Create, edit and delete (typed confirmation) subjects; configure syllabus topics; load clearly labelled demo data |
| Document Library | Two-stage upload (preview, suggestion, explicit confirmation); search, filter, preview, rename, retype, move, replace, bulk actions, delete |
| Question Analysis | Pick papers and notes of the active subject, review and correct extracted questions, set matching options, run, inspect, export |
| Question Explorer | Browse historical questions with source/year/marks, grounded answer outlines and matching evidence; switch to clearly labelled generated practice questions |
| Coverage Lab | Similarity heatmap, section coverage, syllabus-topic coverage, question-to-section links, unmatched questions |
| Compare Papers | Overlap of two or more papers from the same subject (observed history, not a forecast) |
| Study Planner | Read and select source-cited topic notes before analysis; after analysis, use the evidence-backed revision checklist, topic coverage and rules |
| Analysis History | Frozen snapshots; shows whether inputs changed since; compare two runs |
| NLP Lab | Trace preprocessing, vectors, ranking against your notes, segmentation; heuristic demos; list of what is **not** implemented |
| Evaluation | Precision/recall/MRR/MAP for TF-IDF vs Bag-of-Words against ground truth |
| Diagnostics | Live dependency, extraction and storage checks; guidance for blocked libraries |

## How subject isolation works

Isolation is enforced in the backend, not by hiding results in the UI.

```
active subject id
  -> RetrievalService.build_note_corpus()   SQL: WHERE subject_id IN (allowed) AND document_id IN (eligible)
  -> invariant check on every returned section (IsolationViolation if violated)
  -> vectoriser fitted on those sections only
  -> similarity ranking
```

* A missing or unknown subject raises `InvalidSubjectError`. There is no "all documents" fallback.
* A document or paper from another subject raises `CrossSubjectError`.
* Cross-subject analysis needs an explicit switch **and** named extra subjects, and the saved analysis is labelled cross-subject.
* Section IDs are `<document_id>:<n>`, so identical filenames in two subjects cannot collide.
* The vectoriser learns its vocabulary from the notes only, so a section's weights never depend on which paper is loaded.
* Cache keys contain the subject IDs, a corpus fingerprint (document IDs, subject, file hashes, types) and the settings. Moving, replacing or reclassifying a document changes the fingerprint.
* In the UI, every subject-specific session value is stored under a key containing the subject ID, and a saved analysis is displayed only if it belongs to the active subject.

Tests: `tests/test_subject_isolation.py` (backend, including an adversarial case where another subject's notes are a perfect lexical match), `tests/test_ui_analysis.py` and `tests/test_e2e_journey.py` (through the UI).

## NLP methods actually implemented

PDF/PPTX/TXT/MD extraction with page or slide references; Unicode normalisation; tokenisation; stop-word removal (including contractions); optional Porter stemming; question segmentation (numbered, `(a)`, `1(a)`, marks) with manual correction; note segmentation by headings, pages and slides; Bag-of-Words and TF-IDF with unigrams and bigrams; cosine similarity; ranked retrieval; per-term score contributions as explanations; coverage statistics; syllabus-topic coverage; precision@k, recall@k, MRR, MAP; coverage-classification confusion matrix.

**Not implemented** (and not claimed anywhere in the UI): word embeddings or any semantic matching, lemmatisation, trained NER or POS models, OCR. The NLP Lab includes a suffix-rule POS guesser and a 28-term concept dictionary, labelled as illustrative heuristics. Neither is used in retrieval.

## Interpreting results

* Tiers (Strong / Moderate / Weak / Unmatched) are fixed, **uncalibrated** similarity thresholds that you can adjust. They are not confidence levels.
* Sections listed for an *Unmatched* question are the closest by vocabulary and are shown for reference only.
* A topic is **assessable** only if more than half of its words occur in your notes. Otherwise it is *not assessable*, which is different from *not detected* and is not evidence that the topic is absent from exams.
* Subject suggestions are TF-IDF similarities to subject profiles, not probabilities. The filename can only raise a conflict warning.

## Architecture

```
app.py                  page config, theme, navigation, error boundary, degraded mode
components/             theme tokens + CSS, UI primitives, state, sidebar, record-based charts
pages_content/          one module per page (render()), shared helpers
services/               subjects, library, retrieval, classification, analysis runner, history,
                        comparison, topics, study plan, exports, diagnostics, evaluation, overview
nlp/                    preprocessing, segmentation, vectorisation, similarity, heuristics
storage/                SQLite schema + controlled file store (replaceable behind the services)
tests/                  unit, integration and UI (Streamlit AppTest) tests
sample_data/            synthetic demo notes, paper and v2 ground truth
```

Pages never touch SQL or implement NLP; services never import Streamlit. All analysis pages read **saved snapshots**, so history, exports and every view show the same frozen data.

## Data, privacy and security (current state)

* Data lives under `SYLLABUSLENS_DATA_DIR`: a SQLite database and an upload folder. Stored filenames are random IDs; the original filename is metadata only (no path traversal).
* Uploads are validated by extension, size and file header. Extracted text and filenames are HTML-escaped wherever displayed. CSV exports neutralise spreadsheet formula injection.
* Telemetry is disabled in `.streamlit/config.toml`. The app makes no network calls, except that the browser may fetch web fonts from Google Fonts (system fonts are used offline).
* **There is no authentication or authorisation.** Anyone who can reach the app can read, change and delete all data.

## Troubleshooting a blocked SciPy (Windows)

If Windows Application Control, WDAC or AppLocker blocks a native library, the app starts in a limited mode and shows the Diagnostics page with the detected cause. SyllabusLens never bypasses or disables those protections. Ask your administrator to allow the file named in the error, use an already-approved Python distribution in a fresh virtual environment, or run on a machine where the policy does not apply.

## Verification record

| What | Result |
|---|---|
| Automated tests (`python -m pytest -q`) | 180 passed, Linux, Python 3.12 |
| Same suite on the minimum Streamlit (1.50.0, pandas 2.3.3) in a clean venv | 180 passed |
| Real-browser layout audit (Chromium via Playwright; 12 pages at 1440, 1100, 768 and 390 px) | no horizontal overflow, no Streamlit exceptions, no console errors (apart from the offline web-font request) |
| Accessible names, one `h1` per page, `lang` attribute | checked on every page; all reachable controls named |
| Text contrast | all text/background token pairs >= 4.5:1, enforced by `tests/test_design_system.py` |
| Windows | **not tested**; the blocked-library path is simulated, not reproduced |
| Not verified | real-world PDFs from varied sources (tests use generated PDFs), large corpora and performance, screen-reader behaviour, Safari/Firefox, touch devices |

## Known limitations

* Lexical matching: synonyms and paraphrases are missed; shared generic words can create weak false leads.
* Scanned PDFs have no OCR and are flagged "no text".
* Evaluation uses a 10-question synthetic demo set. Its results say nothing about accuracy on other courses.
* Question segmentation depends on conventional numbering; unusual layouts need manual correction on the Question Analysis page.
* The in-process analysis cache is per server process.
* `services/export_service.py` is a legacy exporter for live result objects, superseded by `services/record_export.py`; it is kept because it is tested.

## Before a real deployment

Not done, and required before multi-user or public use: authentication and per-user data ownership (the schema has no user concept); authorisation on every service call; storage with backups, encryption at rest and a retention/deletion policy; a privacy review of uploaded academic material; rate limiting and malware scanning of uploads; a shared session/cache store and scaling plan; HTTPS and security headers via a reverse proxy; monitoring and structured logging; dependency pinning with vulnerability scanning; accessibility testing with assistive technology; Windows and cross-browser testing.
