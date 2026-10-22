# voice-assistant

A local voice assistant: hold a button, ask a question, hear a spoken
answer -- fully on-device, no cloud round trip. A wake word, for hands-free
use, is a choice (see below for why it is not the default).

This brick has almost no model code of its own. It composes three other
bricks -- the same "brick composing bricks" shape `meeting-notes` uses,
just with a third brick added:

- [`live_translation.transcriber`](../live-translation/src/live_translation/transcriber.py)
  for speech-to-text (in `task="transcribe"` mode -- see **A real technical
  finding** below for why that needed a small change upstream).
- [`doc_qa.engine_factory.create_llm`](../doc-qa/src/doc_qa/engine_factory.py)
  for reasoning about what you asked.
- [`voice_clone_studio.voice_model`](../voice-clone-studio/src/voice_clone_studio/voice_model.py)
  for speaking the reply back (in its base voice -- no cloning, no
  reference clip needed).

What is this brick's own is what says a question is being asked.

**By default, a button held while you speak** (since 2026-10-13). In the
launcher it is the panel's **Hold to talk** button, or the space bar; from a
terminal, Enter to start a question and Enter to end it. The button says
where the question begins and ends, so nothing guesses it from the sound,
which is what goes wrong in a loud hall. The third of a second before the
press is kept, so that a word begun as the button goes down is whole; a hold
under 0.4 s is taken for a click and said so; a question stops being
listened to after 28 s. No model is loaded for it.

**Or a wake word**, when one is chosen: **wake-word detection**
([`wake_word.py`](src/voice_assistant/wake_word.py)), via
[openWakeWord](https://github.com/dscripka/openWakeWord), on ONNX Runtime.
It always listens; when it hears the wake word, the rest of the pipeline
(transcribe -> reason -> speak) takes over.

**Why the wake word is not the default: its licence is two licences.**
openWakeWord's code is Apache-2.0; the models it comes with, "hey jarvis"
among them, are CC BY-NC-SA 4.0 -- non-commercial -- by its own README. So
it is there for who may use it, named as such in the menu, and neither
loaded nor fetched until it is chosen. The decision and the other ways out
are in
[docs/MODEL_LICENCES.md](../../docs/MODEL_LICENCES.md#wake-word----hey-jarvis-cc-by-nc-sa-40).

## Why wake-word detection isn't a portable/openvino choice

Every other model-driven piece in this workspace offers a switchable
`portable` vs `openvino` engine, since that's the whole point of the
showcase. Wake-word detection deliberately doesn't: it's a tiny classifier
(three small ONNX models -- melspectrogram, a shared embedding backbone,
and a per-wake-word head) meant to run continuously in the background at
near-zero CPU cost. It isn't the part of a voice assistant that benefits
from NPU/iGPU offload, and OpenVINO's own tooling isn't built around
sub-millisecond always-on classifiers like this one. So `--engine` here
selects the backend for the three *heavy* stages -- transcription,
reasoning, and speech synthesis -- exactly where OpenVINO acceleration
actually matters; wake-word detection runs the same way regardless.

## Setup

From the workspace root (`local_demo/`):

```bash
uv sync                    # portable engine only
uv sync --extra openvino   # also installs the OpenVINO engine
```

> **First run note:** Whisper, the LLM and the TTS voice are downloaded and
> cached on first use (`~/.cache/huggingface`), and so are the wake word's
> models the first time one is chosen (openWakeWord's go to its own package
> directory). Every run after that is fully offline.

## Usage

```bash
uv run voice-assistant --list-devices
```

```bash
uv run voice-assistant
```

Press Enter, ask your question, press Enter again. Press `Ctrl+C` to stop.
With `--wake-word hey_jarvis`: say "Hey Jarvis", then ask.

Listen and answer on the Intel NPU via OpenVINO (the voice itself is always made on the CPU: its model does not
compile for the NPU, and on a GPU it is compiled again for every sentence):

```bash
uv run voice-assistant --engine openvino --compute-device NPU
```

No speakers on this machine? Print replies instead of speaking them:

```bash
uv run voice-assistant --no-speak
```

## Options

| Flag | Description |
| --- | --- |
| `--wake-word {hey_jarvis,alexa,hey_mycroft,hey_rhasspy}` | Listen for this wake word instead of waiting for Enter. Its model is for non-commercial use. Default: none. |
| `--wake-threshold FLOAT` | Detection score threshold (0-1). Lower triggers more easily (and more falsely). Default: `0.5`. |
| `--audio-device NAME` | Substring to match a specific microphone. |
| `--engine {portable,openvino}` | Backend for speech-to-text, the LLM, and text-to-speech. Default: `portable`. |
| `--whisper-model NAME` | Whisper model size override. |
| `--compute-device NAME` | `openvino` engine only: `AUTO`, `CPU`, `GPU`, `NPU` -- for listening and answering. The voice is made on the CPU. |
| `--llm-model {1.5b,8b}` | `openvino` engine only: which language model answers -- Qwen2.5-1.5B (default) or Qwen3-8B. On twelve spoken questions on the NPU (2026-10-10) the 8B took 2.7 s an answer against 0.7 s, before the voice starts; it knew how long to boil an egg where the 1.5B did not, got the same sum wrong, and asked what was decided in yesterday's meeting it made a decision up where the 1.5B asked for details. |
| `--no-speak` | Print replies instead of speaking them out loud. |
| `--list-devices` | List microphones and inference devices, then exit. |

## How it works

[`session.py`](src/voice_assistant/session.py)'s `run()` is the whole loop:

1. Reads the microphone all along. With the button, what it gives is
   dropped until the button goes down (`talk`, an event the caller sets and
   clears), and kept until it comes up. With a wake word, the blocks are fed
   to `WakeWordDetector.triggered()` instead.
2. With a wake word, once triggered, hands the *same* underlying block
   stream to `live_translation`'s energy-based segmenter
   (`pantherlake_ai_core.segmenter.segment_stream`) to capture one command
   utterance -- the same segmenter live-translation itself uses, reused
   as-is, not reimplemented. With the button there is nothing to cut: the
   question is what was held.
3. Transcribes that utterance (`task="transcribe"`, not `"translate"` --
   a voice assistant should hear you in the language you spoke).
4. Asks the LLM for a short, spoken-style answer (1-3 sentences, no
   markdown -- see the system prompt in `session.py`).
5. Synthesizes and plays the reply through this machine's own speakers
   (`pantherlake_ai_core.audio.play`, new this brick -- see below), then
   goes back to listening.

## A real technical finding: stale wake-word buffer state

Verified via a genuinely end-to-end test: synthesizing "Hey Jarvis, what is
two plus two?" with `voice-clone-studio`'s own TTS, feeding it through
`session.run()` with the microphone swapped for that synthetic clip, and
checking the actual sequence of wake/heard/reply events produced. The
first run surfaced a real bug, not a hypothetical one: after answering,
the loop went back to scanning for the wake word and immediately
re-triggered on pure silence.

Cause: while a command is being captured (step 2 above), the wake-word
detector is never called -- capture goes straight to the segmenter
instead. openWakeWord's internal buffer is a rolling window built from
whatever audio it's actually fed, so skipping a stretch of audio like that
leaves its window straddling stale pre-gap content. When scanning resumes,
that stale window can score above threshold on the very next block, even
if that block is silence.

Fixed by calling `WakeWordDetector.reset()` (clears openWakeWord's
prediction and feature buffers) right after every wake -> command cycle,
before resuming the scan loop -- see `session.run()` and
`WakeWordDetector.reset()`'s docstring for the detail. Re-running the same
synthetic test after the fix produced exactly one wake/heard/reply cycle,
not two.

## Notes / current limitations

- **Acoustic loopback testing found a real hardware gap on this
  development machine, not a code issue**: playing synthesized "Hey
  Jarvis" audio through this machine's speakers while the assistant
  listened on its real microphone produced no detection -- a direct mic
  recording taken during playback measured RMS ≈ 0.0004 (effectively
  silence), confirming the mic simply isn't acoustically picking up this
  machine's speaker output, not that detection failed. The pipeline logic
  itself is verified correct via the synthetic block-stream test described
  above (same `session.run()` code, real audio samples, mocked capture
  source only). On a machine where the mic can actually hear the speakers
  (or when spoken to directly), this should work as designed.
- One wake word active at a time -- switching requires restarting the
  session (a new `WakeWordDetector` is constructed at start).
- The energy-based segmenter measures the room's noise floor as it goes,
  from the gaps between words, so a command spoken straight after the wake
  word is heard (it used to need a pause first: the floor was measured
  once, over the first 0.6 s). The first word can still lose its opening
  if there has been no gap at all yet.
- English only, matching `voice-clone-studio`'s current scope.
- **On the NPU, Whisper base can write an English question in French.**
  Seen on 2026-10-13 (OpenVINO 2026.4.1) on three questions said by the
  studio's own voice: "How many days are there in a week?" came back as
  "Comment les jours sont dans le week-end?" on the NPU and word for word on
  the CPU, with the same model. Told that the language is English, the NPU
  reads all three right; this brick does not tell it. Not tried with a
  person speaking. In the backlog's Inbox.
- No barge-in -- you can't interrupt the assistant while it's speaking a
  reply; the next question (a press of the button, or the wake word) is only
  heard after playback finishes.
- **The button was tried without a microphone** (2026-10-13): in the
  launcher, on the NPU, with questions made by the studio's voice put in the
  microphone's place and nothing played -- held with the pointer and with
  the space bar, a click too short, a reload of the page in the middle of a
  run. A person holding it and speaking is still to be heard (backlog R35).
