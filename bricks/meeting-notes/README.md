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

No new code dependencies. On the OpenVINO engine the notes are written by a
model of their own, downloaded the first time notes are asked for (about
5 GB; `uv run panther-lake-prefetch`, or "Models" in the launcher's footer,
fetches it ahead of a show). See **The notes model** below.

## Usage

```bash
uv run meeting-notes --list-devices
```

Transcribe the call/video playing on this device, then generate notes on Ctrl+C:

```bash
uv run meeting-notes --source system
```

On the OpenVINO engine the notes are written on the NPU when the machine has
one (see **Which chip writes the notes**). To have a GPU write them instead,
which gives more complete notes:

```bash
uv run meeting-notes --source system --engine openvino --notes-device GPU
```

The CLI transcribes until you press Ctrl+C, then generates and prints
final notes automatically. The launcher's web UI (`uv run panther-lake-launcher`)
is the better way to watch the transcript grow live and generate notes on
demand at any point, not just at the end.

### From a live translation session

A meeting that [live-translation](../live-translation/README.md) is already
transcribing in the launcher needs no second transcription. **Summarise in
Meeting Notes**, in that panel, hands its whole transcript over and the notes
are written from it here, on the engine and device this panel is set to. It is
refused while this brick is transcribing a meeting of its own.

## Options

| Flag | Description |
| --- | --- |
| `--source {mic,system}` | Audio source. Default: `system` (the call/video itself, not just your mic). |
| `--audio-device NAME` | Substring to match a specific microphone/output device name. |
| `--engine {portable,openvino}` | Backend for *both* transcription and notes generation. Default: `portable`. |
| `--compute-device NAME` | `openvino` engine only: `AUTO`, `CPU`, `GPU`, `NPU`. The chip that transcribes. |
| `--notes-device NAME` | `openvino` engine only: the chip that writes the notes. Default: the NPU if the machine has one, otherwise the same as `--compute-device`. |
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

## The notes model

On the OpenVINO engine the notes are written by **Qwen3-8B**: the standard
int4 build on a GPU or the CPU, and the channel-wise build on the NPU, where
the standard one does not compile. The portable engine keeps `doc-qa`'s small
default (Qwen2.5-1.5B).

Measured on the XPS 14 on 2026-10-07, on four test meetings holding 20 stated
tasks between them -- three written for the purpose, one dictated in French
into the laptop's microphone and transcribed by live translation. One or two
runs per cell; "found" is tasks listed with the right owner and deadline by a
strict pattern check, and every set of notes was read as well.

| Model, chip | Tasks found | Speed | What went wrong |
| --- | --- | --- | --- |
| Qwen2.5-1.5B, integrated GPU (before) | 6 | 84 tok/s | "None identified" on a meeting with four tasks; invented deadlines |
| **Qwen3-8B, integrated GPU** | 17, 18 | 23 tok/s, 6-10 s a meeting | misses "we meet again on..."; one unnamed speaker's task given to the last person mentioned |
| Qwen3-8B, Arc Pro B60 | 17 | 60 tok/s, 2-4 s | the same |
| Qwen3-8B, CPU | (one meeting) | 14 tok/s, 25 s | |
| Qwen3-4B, integrated GPU | 16, 16 | 39 tok/s | two tasks given to the wrong person in every run |
| Qwen3-Coder-30B, integrated GPU | 14-17 | 43 tok/s | no better than the 8B, for 17 GB |
| **Qwen3-8B channel-wise, NPU** | 11, 10 | 19 tok/s, 7-17 s | lists fewer tasks; one garbled name |
| Qwen2.5-1.5B, NPU | 8 | 57 tok/s | wrong owners, placeholder deadlines |
| Mistral-7B-v0.3 channel-wise, NPU | 13 | 20 tok/s | deadlines nobody set ("by the end of the current week"), months added to dates |
| Phi-3.5-mini channel-wise, NPU | (two meetings) | 31 tok/s | invents owners and dates ("John", "Mary", "Friday, October 26th") |
| Qwen3-4B int8, Qwen3-8B int8, NPU | none | | compile in 66 s and 93 s, then no answer in 13 and 8 minutes |

The first load takes about 10 s on a GPU, and about a minute on the NPU the
first time ever (it compiles the model for the chip, then keeps the result).

### Which chip writes the notes

The notes are written on a chip of their own, and left to the app that is
**the NPU**: writing notes is a few seconds of work now and then, which is
what the NPU is for, and it leaves the GPU to the transcription. It also
keeps the speech model and the notes model off the same NPU: two models there
is the situation in which the NPU driver fault was seen (see BACKLOG).
"Notes written on" in the launcher, `--notes-device` here,
choose another chip, and in the launcher the choice can change between two
summaries of the same meeting.

The price is in the table: the NPU's build of the model finds about 11 of the
20 tasks where a GPU finds 17. That build was quantised without calibration
data (its `openvino_config.json` says `"dataset": null`; the standard build's
says `wikitext2`), which is the likely cause, and nothing else published runs
better there -- the rows above are every candidate tried. Asking the NPU's
model for the summary and the action items in two separate calls found 14,
but listed decisions as tasks and none of the three tasks of the one real
transcript, so it was not kept. A channel-wise build quantised *with*
calibration data is what would close the gap; it would have to be made
(`optimum-cli export openvino ... --sym --group-size -1 --awq
--scale-estimation --dataset wikitext2`) and hosted.

## Prompting notes (bugs found while building this)

The instructions ([`session.py`](src/meeting_notes/session.py)) say what a
speech transcript is: one line per utterance and **no speaker names**. The
models gave a task to whoever was mentioned last, so a task is given to a
person only when the transcript makes clear who, and to `Unassigned:`
otherwise -- which is every task of a meeting one person dictated. Three more
things the test meetings taught:

- **A name in an example comes back.** "Maria, can you send it?" as an
  example of a request made every task of a meeting with no Maria in it
  Maria's. The examples name nobody.
- **The model is not shown the time of each line.** With `[10:41:56]` in
  front of every line it wrote dates nobody said ("by today (10/07/2024)"),
  and the times cost tokens the NPU's fixed window needs for the meeting.
- **Plain text.** The page and the terminal show the notes as written, so
  Markdown arrived as asterisks and `#` signs.

Earlier, with the small default LLM (1.5B/int4-scale), the action-items
prompt failed in two different ways before landing on a prompt + guard:

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
- A transcript does not say who is speaking. The notes name a task's owner
  only when the words make it clear, and still get it wrong sometimes: check
  the names before forwarding them.
- On the NPU, which is where the notes go by default, they are noticeably
  weaker than on a GPU (see the table above): the only build of the model
  that runs there loses accuracy. When the notes matter more than the chip,
  set "Notes written on" to a GPU.
- The thin-transcript guard is a word-count heuristic, not a semantic
  check -- it prevents the worst, most obvious hallucination case (near-empty
  input) but doesn't guarantee a longer transcript can't still produce an
  imperfect summary; small local models remain small local models.
- No periodic auto-summarization -- notes are generated only when asked
  for, reflecting the transcript at that moment. Regenerating later
  reflects everything captured since, including earlier notes' source
  material (there's no notes-of-notes chaining).
