# Streaming the language models

Status: steps A and B built and verified on the demo machine on 2026-10-03
(backlog ticket **R32**). Step C, the voice assistant speaking as it writes,
is not built. This note is the design, what was measured, and what the
measurements changed.

Before this, every language-model call in the app blocked until the whole
answer existed. Streaming changes three things a person can see, with one
change underneath:

1. **Live speed.** The hardware panel shows tokens per second while a
   brick is generating, not "last 44 tok/s" once it has finished.
2. **Answers as they are written.** Text appears word by word instead of a
   spinner for the length of the answer.
3. **Stop.** The panel's ✕, or the Stop button beside the text, ends an
   answer part-way. What was written stays on screen, marked incomplete.

It covers the five bricks that answer a request with a language model:
document Q&A, code review, HTML creator, screen OCR and meeting notes.

## What was measured

On the demo machine (Dell XPS 14, Core Ultra X7 358H), OpenVINO GenAI 2026.3.

**It works on every engine and chip.** Qwen2.5-1.5B int4, a 200-token
answer:

| Engine and chip | First text | Stop returns after |
| --- | --- | --- |
| OpenVINO, iGPU | 0.09 s | 0.30 s |
| OpenVINO, NPU | 0.39 s | 0.79 s |
| llama.cpp, CPU | 0.19 s | 0.50 s |

The pieces joined together are exactly the final answer, and a stopped model
answers its next request normally.

**It costs nothing that can be measured.** Interleaved runs of 120 tokens
each (eight per column for the small model, six for the 30B), median tokens
per second:

| Model and chip | No streamer | A plain callback | The token-counting streamer |
| --- | --- | --- | --- |
| Qwen2.5-1.5B, iGPU | 58.3 | 56.0 | 59.0 |
| Qwen2.5-1.5B, NPU | 53.4 | 54.0 | 51.7 |
| Qwen3-Coder-30B, iGPU | 28.8 | 28.8 | 28.6 |

The differences are smaller than the spread between identical runs. An
earlier single pair of runs read as an 8% cost on the iGPU (91.0 against
83.9); that was run-to-run variation, not the streamer. (The iGPU figures
in this note are low: the laptop was on battery at 28% when they were
taken, and earlier the same day the same models gave 87 and 44 tok/s. See
the backlog Inbox.) A streamer running in Python also does not hold up the rest of the
launcher: while one generated, another thread woke about 480 times with a
longest stall of 6 ms, the same as with no streamer.

**A piece of text is not a token.** OpenVINO calls a plain callback only
when a token completes some printable text. For prose that is nearly every
token (199 calls for 206 tokens); for HTML it is not (154 calls for 220
tokens), so a rate counted from callbacks read 30% low on the HTML creator
and code review. The streamer therefore sees every token itself and leaves
the text to OpenVINO's own `TextStreamer`: 220 tokens seen for 220
generated, at a rate that matches the runtime's own figure to the decimal on
both chips.

**Through the launcher**, on battery:

| Brick | Live rate while writing | Final figure | Stop returns after |
| --- | --- | --- | --- |
| Document Q&A, NPU | 34-55 tok/s | 51.5 tok/s | 0.32 s |
| Code review, iGPU (30B) | 25-38 tok/s | 33.0 tok/s | 0.33 s |
| HTML creator, iGPU (30B) | 22-34 tok/s | 32.2 tok/s | 0.33 s |
| Screen OCR, iGPU (7B vision) | 17-20 tok/s | 20.1 tok/s | 0.36 s |
| Meeting notes, NPU, a long meeting in parts | 37-50 tok/s | 46.8 tok/s | 0.30 s |

The live figure follows the last second and a half, so it moves around the
final one, which is the average over the whole answer.

In every case the stopped request came back with the text written so far and
`cancelled: true`, and the model stayed loaded.

## Why it is worth having

- **The wait becomes the demo.** First words arrive in 0.1-0.4 s. Without
  streaming the audience watches a spinner for the whole answer: 5 s for a
  short one, 54 s measured for an HTML page (2,175 tokens).
- **The speed claim becomes live.** A number that moves while text appears
  is the hardware story; a number that shows up afterwards is a caption.
- **A runaway answer can be stopped.** The HTML creator may write 6,144
  tokens; it no longer holds a chip until it finishes.
- **One seam, several later wins**: live time-to-first-token, energy per
  token (R18), speaking a reply sentence by sentence in the voice assistant.

## What to watch

- **Partial output is a state of its own.** Several bricks post-process the
  whole answer -- the HTML creator strips code fences and checks for
  `</html>`, meeting notes drop a contradictory "None identified". None of
  that can run on half an answer, so partial text is shown raw and replaced
  by the finished, processed answer.
- **A stopped answer must look stopped.** Half a code review that reads
  like a finished one is worse than none: it carries a "Stopped -- this
  answer is incomplete" line, and its figures say they describe an
  incomplete answer.
- **Streaming does not shorten the prompt.** A long prompt is read before
  the first token. The page says what the brick is doing until text starts.
- **Not everything streams to the screen.** Expense structuring (JSON for a
  parser), screen-memory OCR and the voice assistant's reply run inside
  loops with their own Stop, and have no reader watching the text; they are
  unchanged.

## How it is built

### 1. One control object through every call (core)

`pantherlake_ai_core.types`:

```python
@dataclass
class GenerationControl:
    on_text: Callable[[str], None] | None = None     # each piece of the answer
    should_stop: Callable[[], bool] | None = None    # asked after every piece
    on_heading: Callable[[str], None] | None = None  # a line the brick adds itself
    on_tokens: Callable[[int], None] | None = None   # how many tokens were just produced
```

`LLM.answer(system, user, max_tokens, control=None)` and the vision-language
extractor take it as one optional argument, so a brick passes a single thing
down. `GenerationStats` has `cancelled: bool`. With no control passed,
nothing changes: no streamer is installed.

- **OpenVINO** (`openvino_streamer`): a `StreamerBase` whose `write` counts
  the token and hands it to a `TextStreamer`; that calls `on_text` and
  answers `CANCEL` when `should_stop()` is true.
- **llama.cpp**: iterate `create_chat_completion(..., stream=True)`, one
  token per chunk, and stop by abandoning the iterator.
- A brick that makes several calls for one answer (code review's two, a long
  meeting's parts) puts its own headings into the text with `say(control,
  ...)` and checks `stopped(control)` before starting the next call.

### 2. The launcher holds the answer in flight

`launcher/generation.py`: a runner calls `generation.get(demo_id).begin()`
and passes the returned control to its brick. That one object keeps the text
so far, reports a live `tok/s` to the hardware panel (over the last 1.5 s,
starting afresh at each heading so the pause before a second answer is not
counted as slow writing), and holds the stop flag.

- `POST /api/bricks/<id>/stop` stops an answer in flight and leaves the
  model loaded; with nothing in flight it unloads the model, as before.
  `?stage=notes` stops meeting notes' summary without ending the
  transcription.
- `GET /api/bricks/<id>/partial` returns `{active, text, cancelled}`.

### 3. The page asks for the text

While its request is pending, the page asks for the partial text every
300 ms and shows it with a Stop button; the finished answer replaces it.
Polling rather than server-sent events or a WebSocket: a missed ask loses
nothing (the next one has more text), and it adds no delivery mechanism
while R05 is open. At 40-55 tok/s the text grows by 40-90 characters per
ask, which reads as live.

### Tests

`tests/test_streaming.py`: stand-in runtimes for both backends; a stop
part-way returns the partial text and `cancelled=True`; tokens are counted
apart from pieces of text; code review never starts its second answer after
a stop; a long meeting keeps the parts already done; the stop route stops an
answer first and unloads second. The page was checked by hand in the
browser.

## Decided

1. Steps A and B together (2026-10-03).
2. Partial text is polled.
3. A stopped answer stays on screen, labelled incomplete.

## Not built

- **Step C**: the voice assistant speaking its reply sentence by sentence
  as it is written (1-2 days).
- A live rate and a mid-answer stop for the language-model calls inside the
  loop bricks (expense extraction, screen memory, voice assistant).
