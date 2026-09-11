# Sample data

## Current demo kit — Meridian customer rollout, September 2026

Everything in the new kit is fictional and marked accordingly. Vendors, people,
addresses, reference numbers, documents and transactions are invented. No real
logos, payment credentials, third-party voices or private files are included.

| Brick | Few-click path | Bundled content |
| --- | --- | --- |
| Expense Report Extractor | Open → choose pack → Start → review → Export approved CSV | 3-receipt quick start, 9-receipt trip, 3 review cases; editable fields, previews and answer key |
| Screen / Image Text Extraction | Open → choose sample → Extract text | Hotel invoice, cafe receipt and refund; no file-picker hunt |
| Document Q&A | Open → choose question → Index folder → Ask | Six linked documents covering release gates, customer scope, incidents and money |
| HTML Creator | Open → choose prompt → Generate | Source-grounded launch brief, operating dashboard or fictional boutique hotel |
| Code Review | Open → choose diff → Review | Tenant export regression and incomplete map-validation fix; never executed |
| Voice Clone | Open → enroll own voice → choose script → synthesize | Handover, metrics briefing and hotel welcome |
| Screen Memory | Open guide → show demo page while recording → stop → search | A readable fictional operating brief; no prebuilt search result |
| Translation / Meeting Notes / Voice Assistant | Open → expand Demo guide → read script into chosen microphone | French customer update, meeting stand-up and practical questions |
| Vision demos | Open → expand Demo guide | Honest live-camera scenarios; real online city feeds remain explicitly real |

The controls may first download their models. Opening a sample never starts a
capture, model run, payment or network video feed by itself. Each brick has a
Demo guide, and selected file-based samples show their source previews/links.

- `expenses/`: deterministic receipt images, overview and independent answer key.
  Expected full-trip totals: **EUR 1,048.20 and USD 49.00 separately**. Quick start:
  **EUR 215.10**. Review cases must remain outside validated totals.
- `meridian-rollout-2026/`: six source documents, less than the HTML input budget.
  Ground-truth hints stay in guides, not in the indexed source folder.
- `review-diffs/`: invented changes for review, not patches to apply.
- `recall-demo.html`: local, self-contained screen-memory target; no external assets.
- `catalog.json` and `guides.json`: shared prompt and guide source of truth.

Rebuild receipt PNGs with `python scripts/build_demo_samples.py` (Pillow required).
The script checks line-item arithmetic and generates the contact sheet. Expected
results describe the fixtures; they do not claim that a particular model produced
those results. The parser remains free to flag uncertain OCR for review.

## Legacy samples

The older folders below are retained for compatibility with existing local paths.
The current pickers use the revised kit above. The new document folder also avoids
reusing a stale vector index created from the older Meridian material.

Fictional demo content, checked in so bricks that ingest a folder of
documents (`doc-qa`, and `html-creator`'s document-summary mode) have
something real to run their sample questions/prompts against. Not real
company data -- company name, people, numbers, and events below are all
invented for demo purposes.

- `meridian-robotics/` -- a fictional warehouse-robotics startup: a
  company overview, product roadmap, a meeting-notes doc, an HR policy,
  and a board update. Deliberately cross-referential (the same battery
  issue and safety incident show up in the roadmap, the meeting notes,
  and the board update) so multi-document questions have something to
  actually connect.
- `receipts/` -- three synthetic till receipts (a cafe, a taxi ride, an
  office-supply store) rendered as PNGs for `expense-extract`'s "Try a
  sample" picker. Fictional vendors and amounts; generated, not
  photographed, so there is no real card, address, or person in them.
