# smart-recall

Semantic search over your own screen history -- fully local. Periodically
captures the screen, reads what's on it, and indexes it, so you can later
ask "that pricing page from yesterday" and get the exact screenshot back.
The "Windows Recall" idea, minus sending your screen history anywhere.

Like `expense-extract`, this is a **genuinely concurrent two-device
pipeline**, not a one-engine-at-a-time demo: OCR ([`screen-ocr`](../screen-ocr))
and embedding ([`doc-qa`](../doc-qa)'s embedder) run on separate threads,
each pinned to an independently chosen device -- while a capture is being
embedded and indexed on one device, the *next* capture is already being
OCR'd on the other. Here that pipeline runs continuously in the
background for as long as recording is on, rather than over a finite
batch of files.

## Setup

From the workspace root (`local_demo/`):

```bash
uv sync                    # portable engines only
uv sync --extra openvino   # also installs the OpenVINO engines
```

## Usage

```bash
uv run smart-recall list-devices
```

Record in the background (Ctrl+C to stop):

```bash
uv run smart-recall record
```

The actual point of this brick -- OCR on the GPU, embedding on the NPU,
**at the same time** (see [screen-ocr's README](../screen-ocr/README.md)
for why GPU, not NPU, is the OCR device that's actually proven working):

```bash
uv run smart-recall record --ocr-engine openvino --ocr-device GPU --embed-engine openvino --embed-device NPU
```

Search everything recorded so far:

```bash
uv run smart-recall search "that pricing page"
```

## Options

`smart-recall` has two subcommands with different flags, since recording
and searching are genuinely different operations:

**`record`**

| Flag | Description |
| --- | --- |
| `--screen-index N` | Screen/monitor index. Default: `1`. |
| `--interval SECONDS` | Seconds between capture attempts -- a capture is skipped entirely if the screen hasn't visibly changed. Default: `5.0`. |
| `--ocr-engine {portable,openvino}` | Backend for the OCR stage. Default: `portable`. |
| `--ocr-device NAME` | `openvino` OCR engine only: `AUTO`, `CPU`, `GPU`, `NPU`. |
| `--embed-engine {portable,openvino}` | Backend for the embedding stage. **Fixed for the life of the index** -- see below. Default: `portable`. |
| `--embed-device NAME` | `openvino` embed engine only: `AUTO`, `CPU`, `GPU`, `NPU`. |
| `--reset` | Wipe any existing index and saved screenshots before starting. |

**`search`**

| Flag | Description |
| --- | --- |
| `question` | What you're looking for, in plain language. |
| `--top-k N` | Number of matches to show. Default: `5`. |
| `--embed-device NAME` | `AUTO`, `CPU`, `GPU`, or `NPU` -- the index's own embedding *engine* is used automatically (see below), only the device is a free choice here. |

## A real technical finding: the embedding engine can't be a per-search choice

`doc-qa`'s `portable` embedder (nomic-embed-text via llama.cpp) and its
`openvino` embedder (Qwen3-Embedding-0.6B via OpenVINO) are two genuinely
different models producing vectors in two different, incompatible spaces
-- not just two runtimes for the same weights, the way this workspace's
OCR/LLM engine pairs usually are. Searching with one while the index was
built with the other doesn't degrade gracefully; the similarity scores
would just be meaningless. So the embedding *engine* is decided once, the
first time you record, and locked in (`index/meta.json`) -- `search`
reads it automatically rather than taking it as a flag, and `record`
refuses to continue an existing index with a different `--embed-engine`
until you `--reset`. Only the compute *device* (which chip runs that same
model) stays a free choice at both record and search time.

## A real bug found through testing: one bad frame used to wedge the whole pipeline

Testing this against a live capture (not synthetic data) surfaced a real
crash: OCR picked up a non-ASCII character, and printing it in the CLI's
event callback raised `UnicodeEncodeError` on a Windows console's default
codepage. That exception propagated out of the indexing thread entirely
-- the thread died, but the OCR thread had no idea and kept enqueueing
captures behind a now-unconsumed bounded queue, which deadlocks the whole
pipeline once the queue fills (`ocr_thread.join()` in `pipeline.run()`
would then never return).

Two fixes, not one, because either alone would leave a real gap:

- `cli.py`'s `main()` now reconfigures stdout/stderr to UTF-8 with
  `errors="replace"` -- OCR'd text can contain any script, any symbol,
  and a CLI that crashes on it instead of just displaying it isn't
  acceptable for a brick whose entire job is showing you arbitrary screen
  content back.
- `pipeline.py`'s indexing worker now catches exceptions **per item**,
  not around the whole thread -- a bad frame (or a misbehaving callback)
  gets reported as a `skipped` event and the loop continues, instead of
  taking the whole background thread down silently.

## How it works

[`pipeline.py`](src/smart_recall/pipeline.py):

1. `run()` starts two daemon threads sharing a bounded `queue.Queue`
   (size 3):
   - **Capture + OCR worker**: every `--interval` seconds, grabs a
     screenshot, checks it against the previous one
     ([`change_detection.py`](src/smart_recall/change_detection.py) --
     a cheap downsampled-grayscale mean-abs-diff, so a static desktop
     doesn't get re-indexed on every tick), and if it changed and OCR
     found readable text, puts `(frame, text)` on the queue.
   - **Embed + index worker**: pulls each item off the queue, saves the
     frame as a JPEG, embeds the text (pinned to `--embed-device`), and
     appends it to the persisted vector index.
2. `RecallIndex` is the read side: loads the persisted index and *only*
   the embedder needed for a query -- no OCR, no screen capture -- and
   exposes `.search(question, top_k)`.

`doc_qa.store.VectorStore` gained a genuinely reusable `.add(chunks,
vectors)` method for this brick -- an additive, backward-compatible
change (its existing `.build()` still fully replaces the index in one
shot, for `doc-qa`'s own folder-ingest use case); `.add()` is for
incremental/streaming ingestion instead, appending to whatever's already
there.

## Notes / current limitations

- **This is a demo, not a production screen-recording tool.** It has no
  sensitive-content filtering -- no password-field detection, no
  excluded-app list. Everything visible gets OCR'd and indexed, in a
  local cache under `~/.cache/pantherlake-ai-studio/smart-recall/`,
  exactly like it sounds. Know what's on your screen before you turn
  recording on.
- Change detection is a global frame diff, not region-aware -- a small
  UI update in the corner of a large screen may fall under the
  0.02 threshold and get skipped, same as a genuinely unchanged screen.
- No automatic cleanup -- the index and screenshots grow until you
  `--reset` them by hand.
- English OCR/embedding only, matching this workspace's other text bricks.
