# expense-extract

Batch-converts a folder of receipt photos into a CSV of structured expense
lines (vendor, date, amount, category) -- fully on-device.

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
