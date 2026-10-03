# Streaming the language models: groundwork

Status: design, not built. Backlog ticket **R32**. Written 2026-10-03.

Today every language-model call in the app blocks until the whole answer
exists. Streaming changes three things a person can see, with one change
underneath:

1. **Live speed.** The hardware panel shows tokens per second while a
   brick is generating, not "last 44 tok/s" once it has finished.
2. **Answers as they are written.** Text appears word by word instead of a
   spinner for the length of the answer.
3. **Cancel.** The panel's ✕ stops a brick mid-answer. Today it is disabled
   until the answer is done.

## What was measured

On the demo machine (Dell XPS 14, Core Ultra X7 358H), Qwen2.5-1.5B int4,
a 200-token answer, OpenVINO GenAI 2026.3:

| Engine and chip | Without a streamer | With a streamer | First text | Cancel returns after |
| --- | --- | --- | --- | --- |
| OpenVINO, iGPU | 91.0 tok/s | 83.9 tok/s (-8%) | 0.09 s | 0.30 s |
| OpenVINO, NPU | 53.9 tok/s | 55.8 tok/s (no cost) | 0.39 s | 0.79 s |
| llama.cpp, CPU | -- | streams | 0.19 s | 0.50 s |

What that establishes:

- **It works on every engine and chip the app uses**, the NPU included. The
  pieces joined together are exactly the final answer.
- **A cancelled model is still usable.** After a cancel the same pipeline
  answered the next request normally, on both chips, and the text written
  so far is returned.
- **The callback fires about once per token** (189 times for 200 tokens: the
  runtime holds a piece back until it forms printable text), so counting
  callbacks is a fair live rate, and the exact count still comes from the
  runtime's own metrics at the end.
- **The vision-language and Whisper pipelines take the same streamer**, so
  the screen-OCR model and speech are not a separate problem.

OpenVINO's streamer is a callable given each piece of text; it answers
`StreamingStatus.RUNNING`, `STOP` or `CANCEL`. llama.cpp streams with
`stream=True` and is stopped by abandoning the iterator.

## For

- **The wait becomes the demo.** First words arrive in 0.1-0.4 s. Without
  streaming the audience watches a spinner for the whole answer: 5 s for a
  short one, 54 s measured for an HTML page (2,175 tokens).
- **The speed claim becomes live.** A number that moves while text appears
  is the hardware story; a number that shows up afterwards is a caption.
- **Every brick can be stopped.** A runaway generation (the HTML creator may
  write 6,144 tokens) no longer holds a chip until it finishes, and the
  panel's ✕ means the same thing everywhere.
- **It is cheap.** At most 8% of throughput on the iGPU, nothing measurable
  on the NPU.
- **One seam, several later wins**: live time-to-first-token, energy per
  token (R18), speaking a reply sentence by sentence in the voice assistant.

## Against, and what to watch

- **It touches every language-model path**: two LLM backends, the
  vision-language extractor, seven bricks' session APIs, and a way to carry
  partial text for the five bricks that answer a plain request.
- **Partial output is a new state.** Several bricks post-process the whole
  answer -- the HTML creator strips code fences and checks for `</html>`,
  expense extraction parses JSON, meeting notes drop a contradictory "None
  identified". None of that can run on half an answer: partial text has to
  be shown raw and replaced by the finished, processed answer.
- **Not everything should stream to the screen.** Expense structuring (JSON
  for a parser) and screen-memory OCR have no reader watching; they want
  the live rate and cancel, not the text.
- **Streaming does not shorten the prompt.** A long prompt is read before
  the first token, and the NPU's first text already takes 0.39 s on a short
  one. A long meeting still starts with a pause; the UI should say "reading
  the transcript" rather than look hung.
- **A cancelled answer must look cancelled.** Half a code review that reads
  like a finished one is worse than none: label it.
- **More moving parts in delivery.** Event delivery between launcher and
  page is already the open reliability ticket (R05). The transport chosen
  below is deliberately the simplest one for that reason.
- **Testing**: backends need streaming fakes; the page behaviour can only be
  checked by hand.

## How to build it

### 1. One control object through every call (core)

```python
@dataclass
class GenerationControl:
    on_text: Callable[[str], None] | None = None   # each piece as it arrives
    should_stop: Callable[[], bool] | None = None  # asked once per piece
```

`LLM.answer(system, user, max_tokens, control=None)` and the
vision-language extractor take it as one optional argument, so the seven
bricks pass a single thing down rather than growing two parameters each.
`GenerationStats` gains `cancelled: bool`.

- **OpenVINO**: the streamer is a closure -- call `on_text`, return `CANCEL`
  when `should_stop()` is true, else `RUNNING`.
- **llama.cpp**: iterate `create_chat_completion(..., stream=True)`, call
  `on_text`, break when `should_stop()`.

With no control passed, nothing changes: no streamer is installed, and the
8% is not paid.

### 2. The launcher: a cancel flag and a live rate per brick

- Each one-shot runner owns a `threading.Event`. `POST
  /api/bricks/<id>/stop` sets it while a request is in flight (and still
  unloads the model when the brick is idle), so `can_stop` becomes true for
  every brick and the panel's ✕ is never disabled.
- The runner's `on_text` ticks a `metrics.RateMeter`, reporting a live
  `tok/s` -- the panel already draws whatever is reported, so the live
  number needs no front-end change.

### 3. Getting partial text to the page: poll first

Three ways to carry the text of a request/response brick:

| | How | Cost |
| --- | --- | --- |
| **Poll** | the runner keeps `partial_text`; the page asks `GET /api/<id>/partial` a few times a second while its request is pending | no new transport; the same "latest wins" shape as the video feeds |
| Server-sent events | one streaming response per brick | reconnect and ordering to handle |
| WebSocket | as the streaming bricks use | most plumbing; duplicates R05's concerns |

Polling first. Text at 3-4 updates a second reads as live, a missed poll
loses nothing (the next one has more), and it adds no delivery mechanism
while R05 is open. Server-sent events are the upgrade if it ever feels
coarse.

### 4. The page

`Panel.run` accepts where partial text goes; it shows it with a caret while
the request is pending and replaces it with the finished answer. A stopped
answer keeps its text under a "Stopped -- incomplete" label.

### 5. Order, and what each step delivers

| Step | Delivers | Size |
| --- | --- | --- |
| **A** | control object in both backends and the extractor; cancel flag and route; live `tok/s` in the panel; ✕ enabled everywhere | about 1 day |
| **B** | answers word by word in document Q&A, code review, HTML creator (raw until complete), screen OCR, and meeting notes' final merge (parts report "part 2 of 4" as now) | about 2 days |
| **C** (optional) | the voice assistant speaks a reply sentence by sentence as it is written | 1-2 days |

Step A alone delivers two of the three visible changes with no transport
work, and is the natural first commit.

### Not streamed to the screen

Expense structuring, screen-memory OCR and embeddings: live rate and cancel
only.

### Tests

A fake LLM that yields pieces; cancel mid-way returns the partial text and
`cancelled=True`; the live rate is reported and cleared; the stop route
sets the flag for a busy brick and unloads an idle one; with no control
passed, behaviour is byte-for-byte what it is today.

## To decide

1. Step A, then B -- or A only for now?
2. Polling for partial text (recommended) or server-sent events from the
   start?
3. Should a stopped answer be kept on screen, labelled, or cleared?
