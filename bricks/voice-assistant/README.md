# voice-assistant

A hands-free local voice assistant: say a wake word, ask a question, hear a
spoken answer -- fully on-device, no cloud round trip.

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

The one genuinely new piece is **wake-word detection**
([`wake_word.py`](src/voice_assistant/wake_word.py)), via
[openWakeWord](https://github.com/dscripka/openWakeWord) (Apache-2.0 /
MIT-licensed components, ONNX Runtime). It always listens; when it hears
the wake word, the rest of the pipeline (transcribe -> reason -> speak)
takes over.

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

> **First run note:** Whisper, the LLM, the TTS voice, and the wake-word
> models are all downloaded and cached on first use (mostly
> `~/.cache/huggingface`; openWakeWord's models go to its own package
> directory). Every run after that is fully offline.

## Usage

```bash
uv run voice-assistant --list-devices
```

```bash
uv run voice-assistant
```

Say "Hey Jarvis", pause briefly, then ask your question. Press `Ctrl+C` to
stop.

Run every heavy stage on the Intel NPU via OpenVINO:

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
| `--wake-word {hey_jarvis,alexa,hey_mycroft,hey_rhasspy}` | Wake word to listen for. Default: `hey_jarvis`. |
| `--wake-threshold FLOAT` | Detection score threshold (0-1). Lower triggers more easily (and more falsely). Default: `0.5`. |
| `--audio-device NAME` | Substring to match a specific microphone. |
| `--engine {portable,openvino}` | Backend for speech-to-text, the LLM, and text-to-speech. Default: `portable`. |
| `--whisper-model NAME` | Whisper model size override. |
| `--compute-device NAME` | `openvino` engine only: `AUTO`, `CPU`, `GPU`, `NPU`. |
| `--no-speak` | Print replies instead of speaking them out loud. |
| `--list-devices` | List microphones and inference devices, then exit. |

## How it works

[`session.py`](src/voice_assistant/session.py)'s `run()` is the whole loop:

1. Continuously feeds microphone blocks to `WakeWordDetector.triggered()`.
2. Once triggered, hands the *same* underlying block stream to
   `live_translation`'s energy-based segmenter
   (`pantherlake_ai_core.segmenter.segment_stream`) to capture one command
   utterance -- the same segmenter live-translation itself uses, reused
   as-is, not reimplemented.
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
- The energy-based segmenter recalibrates its noise floor for ~0.6s at the
  start of every command capture. Pause briefly after the wake word before
  speaking, the same way you naturally would with any wake-word assistant
  -- speaking immediately risks the calibration window capturing your
  voice instead of silence, throwing off the speech-detection threshold.
- English only, matching `voice-clone-studio`'s current scope.
- No barge-in -- you can't interrupt the assistant while it's speaking a
  reply; the next wake word is only heard after playback finishes.
