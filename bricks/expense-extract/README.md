# expense-extract

Batch-converts a folder of receipt photos into a CSV of structured expense
lines (vendor, date, amount, currency, category, review status) -- fully on-device.

## Trustworthy output

Amounts use decimal arithmetic. `12,50 EUR` becomes `12.50`, not `1250`;
ambiguous strings such as `1,234` need review. Currency must be supported and
present in the OCR text (an explicit code or an unambiguous euro/pound symbol).
A bare `$` does not establish USD. Missing/invalid dates, unsupported currency,
ambiguous amounts and amounts not found in the OCR text are flagged.

Only lines passing these field checks contribute to totals, grouped by currency.
These checks do not verify that the model selected the correct receipt total;
compare results with the originals. The launcher now provides an in-app
correction and approval workspace; its totals include only human-approved rows.

The CSV now includes `currency`, `needs_review` and `review_reasons`. WebSocket
amounts are exact decimal **strings** (or null), and completion messages contain
a `totals` object keyed by currency instead of a single `total`. CLI and UI use
the same extraction validation. Launcher review/export adds explicit human
approval after extraction. Update custom consumers of the old format.

This is the workspace's first brick where **two heavy models genuinely
run at the same time on two different pieces of silicon**, not one
engine choice applied to everything. OCR ([`screen-ocr`](../screen-ocr))
and LLM structuring ([`doc-qa`](../doc-qa)'s LLM) run on separate
threads connected by a small bounded queue -- while the LLM is
structuring receipt *N*, OCR is already reading receipt *N+1*. Pin OCR to
the GPU and structuring to the NPU (or CPU/GPU, or any other pairing) and
both are genuinely busy at once, which is the entire point of this brick:
proving a "complex" workload can actually use more than one accelerator
concurrently, not just switch between them.

## Setup

From the workspace root (`local_demo/`):

```bash
uv sync                    # portable engines only
uv sync --extra openvino   # also installs the OpenVINO engines
```

> **First run note:** OCR and LLM models are downloaded and cached on
> first use (mostly `~/.cache/huggingface`). The `openvino` OCR engine
> specifically is a vision-language model (see `screen-ocr`'s README for
> why RapidOCR's own OpenVINO backend can't target NPU/GPU) -- its first
> compile for a given device, especially NPU, is the slowest part of a
> first run by far. Every run after that is fully offline and fast.

## Usage

```bash
uv run expense-extract --list-devices
```

Process a folder with both stages on CPU (works everywhere, no OpenVINO
needed):

```bash
uv run expense-extract ./receipts
```

The actual point of this brick -- OCR on the GPU, structuring on the NPU,
**at the same time**:

```bash
uv run expense-extract ./receipts --ocr-engine openvino --ocr-device GPU --llm-engine openvino --llm-device NPU
```

Watch the timestamps in the output: OCR lines for several receipts appear
before the LLM has finished structuring the first one -- that gap is the
queue letting the faster stage run ahead, proof the two stages are
actually overlapped rather than strictly alternating.

## Options

| Flag | Description |
| --- | --- |
| `folder` | Folder of receipt image files (`.png`/`.jpg`/`.jpeg`/`.bmp`/`.tif`/`.webp`). |
| `--output PATH` | Output CSV path. Default: `expenses.csv`. |
| `--ocr-engine {portable,openvino}` | OCR backend. Default: `portable`. |
| `--ocr-device NAME` | `openvino` OCR engine only: `AUTO`, `CPU`, `GPU`, `NPU`. |
| `--llm-engine {portable,openvino}` | LLM structuring backend. Default: `portable`. |
| `--llm-device NAME` | `openvino` LLM engine only: `AUTO`, `CPU`, `GPU`, `NPU`. |
| `--list-devices` | List available inference devices, then exit. |

`--ocr-engine`/`--ocr-device` and `--llm-engine`/`--llm-device` are
**independent** -- that's deliberate, not an oversight. Every other
switchable-backend brick in this workspace has one `--engine` flag
because it drives one model; this brick drives two, and the whole demo
is about pointing them at two different places at once.

## How it works

[`pipeline.py`](src/expense_extract/pipeline.py)'s `run()` is the whole
thing:

1. Lists the receipt images in the given folder.
2. Starts two daemon threads sharing a bounded `queue.Queue` (size 3 --
   enough for a faster OCR stage to get ahead of a slower LLM stage
   without unbounded memory growth):
   - **OCR worker**: loads each image, runs it through
     `screen_ocr.pipeline.OcrSession` (pinned to `--ocr-device`), puts
     `(filename, raw_text)` on the queue.
   - **LLM worker**: pulls each `(filename, raw_text)` off the queue,
     asks `doc_qa.engine_factory.create_llm`'s LLM (pinned to
     `--llm-device`) to structure it into JSON
     ([`parsing.py`](src/expense_extract/parsing.py) tolerates the ways a
     small local model deviates from "JSON only" -- markdown fencing,
     a stray sentence before or after), and appends an `ExpenseLine`.
3. Joins both threads, returns every `ExpenseLine` (in completion order,
   which isn't necessarily file order, since the two stages overlap).

Neither worker knows or cares what the other is doing beyond the queue --
that's what makes the concurrency real rather than simulated.

## A real technical finding: why GPU-for-OCR / NPU-for-LLM, not the other way

The first version of this brick's flagship example paired OCR on the NPU
with structuring on the GPU. Testing that pairing found `screen-ocr`'s
OpenVINO engine -- a 7B-parameter vision-language model -- reliably fails
to *compile* for NPU on this hardware, with a compiler-level error
(`[vpux-compiler] UnrollDistributedOps Pass failed: Can't convert 76 Bit
to Byte`), not a runtime fluke. `doc-qa`'s LLM, much smaller, compiles and
runs on NPU without issue. So the pairing that's actually verified
working -- and what this brick, and the launcher's auto-selected
defaults, now use -- is **OCR on GPU, structuring on NPU**: the reverse
of the first instinct, because the bigger model is the one that needs the
device with fewer NPU-compiler edge cases. See
[`screen-ocr`'s README](../screen-ocr/README.md#screen-ocr) for the full
finding -- it's a `screen-ocr` limitation, surfaced by this brick, not
something wrong with this brick's own pipeline code.

## Notes / current limitations

- If OCR can't read any text from a receipt, that line is recorded with
  `error="No text detected by OCR"` rather than silently dropped or
  crashing the batch -- check the CSV's `error` column.
- If the LLM's reply can't be parsed as JSON even after `parsing.py`'s
  fallbacks, same treatment: `error="Could not parse a structured
  response from the LLM"`, raw OCR text preserved in the CSV so nothing
  is lost, just not automated for that one receipt.
- Category is one of a fixed small set (`Meals`, `Travel`, `Lodging`,
  `Office Supplies`, `Software`, `Other`) the LLM is asked to pick from --
  there's no learned/configurable taxonomy.
- Results come back in completion order, not file order (a natural
  consequence of two threads racing) -- the CSV is written in whatever
  order `run()` returns, not alphabetical by filename.
## Launcher review workflow

Choose a sample or receipt folder and press **Start**. When extraction finishes
(or is stopped), the review workspace shows each original receipt beside editable
vendor, date, amount, currency and category fields. Original extraction and OCR
text remain available in the disclosure below the editor.

- **Save pending** keeps corrections without approving the expense.
- **Approve & next** validates the fields, includes the receipt in currency-specific
  totals, and advances to the next pending receipt. Refunds use negative amounts.
- **Exclude** removes a receipt from totals and export; it can be reopened later.
- Identical image bytes or matching vendor/date/amount/currency produce a possible
  duplicate warning. Exclude the duplicate or record why it is a separate expense.
  Newly introduced matches reopen prior approvals for review. This is a heuristic,
  not a guarantee that every duplicate is detected.
- **Export approved CSV** exports only approved records and their review notes.
  Pending and excluded counts remain visible. Text cells are escaped against
  spreadsheet formula interpretation; numeric refund amounts remain numeric.

The latest batch, corrections and original extraction are saved atomically in
`logs/expense-review.json` (ignored by Git), including local image paths and OCR
text. Refreshes and launcher restarts retain the review; interrupted extraction
is labeled accordingly. Source images stay in their original folder and must
remain unchanged for preview, approval and export. Starting a new batch replaces
the saved review after a browser confirmation; export before replacing it.
This is a single-operator, single-batch workspace, not a multi-user approval system.
The CLI retains its existing extraction CSV behavior; human approval applies to
the launcher review/export flow.
