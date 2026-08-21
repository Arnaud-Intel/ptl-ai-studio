# meeting-notes

Live-transcribes a call/meeting and generates a running summary + action
items with a local LLM — fully on-device.

This brick is different from every other one in this workspace: it has
**no transcriber and no LLM of its own**. It imports
[`live_translation.pipeline`](../live-translation/src/live_translation/pipeline.py)
for capture→segment→transcribe, and
[`doc_qa.engine_factory.create_llm`](../doc-qa/src/doc_qa/engine_factory.py)
for the LLM that turns a transcript into notes. Writing a third Whisper
wrapper or llama.cpp wrapper here would just be a bug generator with extra
steps -- the workspace's own README says bricks can depend on each other,
and this is that, for real.

## Setup

From the workspace root (`local_demo/`):

```bash
uv sync                    # portable engine only
uv sync --extra openvino   # also installs the OpenVINO engine
```

No new heavy dependencies -- if you've already used `live-translation` and
`doc-qa`, their models are already cached and this brick has nothing left
to download.

## Usage

```bash
uv run meeting-notes --list-devices
```

Transcribe the call/video playing on this device, then generate notes on Ctrl+C:

```bash
uv run meeting-notes --source system
```

Same, but transcription and notes generation both run on the NPU:

```bash
uv run meeting-notes --source system --engine openvino --compute-device NPU
```

The CLI transcribes until you press Ctrl+C, then generates and prints
final notes automatically. The launcher's web UI (`uv run panther-lake-launcher`)
is the better way to watch the transcript grow live and generate notes on
demand at any point, not just at the end.

## Options

| Flag | Description |
| --- | --- |
| `--source {mic,system}` | Audio source. Default: `system` (the call/video itself, not just your mic). |
| `--audio-device NAME` | Substring to match a specific microphone/output device name. |
| `--engine {portable,openvino}` | Backend for *both* transcription and notes generation. Default: `portable`. |
| `--compute-device NAME` | `openvino` engine only: `AUTO`, `CPU`, `GPU`, `NPU`. |
| `--whisper-model NAME` | Whisper model size override. |
| `--list-devices` | List microphones, output devices, and inference devices, then exit. |

## How it works

[`session.py`](src/meeting_notes/session.py)'s `MeetingSession` is the
whole brick:

1. `transcribe()` calls `live_translation.pipeline.run(...)`, appending
   each returned utterance to a running transcript and forwarding it to a
   caller-supplied callback (so the CLI can print it live and the launcher
   can stream it over a WebSocket -- see `live-translation`'s own README
   for why that split exists).
2. `generate_notes()` takes everything transcribed so far, and asks a
   `doc_qa` LLM (loaded lazily -- no reason to pay for it if notes are
   never requested) to produce a short summary and an action-items list,
   via one deliberately explicit system prompt (see **Prompting notes**
   below). Callable independently of `transcribe()` at any point, including
   after the session has stopped -- the transcript and the LLM both outlive
   the capture thread.

## Prompting notes (a real bug found while building this)

The first version of the action-items prompt was reasonable-sounding but
under-specified, and the small default LLM (1.5B/int4-scale) failed in two
different ways before landing on the current prompt + guard:

- **Under-detection**: a plain "list any action items" instruction missed
  first-person commitments like "I still need to write the tests by
  Friday" entirely, reporting "None identified" on a transcript that
  clearly had two. Fixed by explicitly telling the model that first-person
  commitments and stated timeframes count as action items, not just
  sentences that look like an explicit task list.
- **Hallucination on thin input**: tested against a short, unrelated
  2-line transcript ("nice weather today" / "yeah, sunny this week"), the
  model didn't say there was nothing to summarize -- it invented three
  named attendees and three fake action items wholesale. A stronger "don't
  invent things" instruction did not reliably stop this. Small local
  models cannot be trusted to reliably refuse ungroundable input on
  instruction alone, so `generate_notes()` has a **deterministic** guard
  instead: below `_MIN_WORDS_FOR_NOTES` (25) words of transcript, it
  raises before the LLM is even called -- no LLM call, no chance to
  hallucinate. This is why "Generate notes" can return an error early in a
  meeting; that's working as intended, not a bug.

## Notes / current limitations

- One engine choice drives both transcription and notes generation. You
  can't currently mix e.g. OpenVINO Whisper with the portable LLM.
- The thin-transcript guard is a word-count heuristic, not a semantic
  check -- it prevents the worst, most obvious hallucination case (near-empty
  input) but doesn't guarantee a longer transcript can't still produce an
  imperfect summary; small local models remain small local models.
- No periodic auto-summarization -- notes are generated only when asked
  for, reflecting the transcript at that moment. Regenerating later
  reflects everything captured since, including earlier notes' source
  material (there's no notes-of-notes chaining).
