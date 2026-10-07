# live-translation

A small local tool that listens to audio and prints its **English translation**
as text, in near real time — entirely on-device, no audio or text ever
leaves the machine. This brick is part of the
[Panther Lake local AI demos](../../README.md) workspace.

It can listen to either:

- your **microphone**, or
- your device's **system audio output** (loopback) — e.g. a video, call, or
  stream currently playing — with no "Stereo Mix" device required.

It supports two interchangeable inference engines:

- **`portable`** (default) — [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
  (CTranslate2). Runs anywhere: CPU, or CUDA on an NVIDIA GPU.
- **`openvino`** — [OpenVINO](https://docs.openvino.ai/) via `openvino_genai`.
  Targets Intel hardware explicitly: `CPU`, `GPU` (iGPU), or `NPU` — this is
  what actually showcases Panther Lake's on-device acceleration rather than
  just running on any laptop's CPU.

Either way, Whisper's `translate` task automatically detects the spoken
language and translates it straight to English, so this works for speech in
essentially any language Whisper supports.

Detection is done afresh for every utterance, from that utterance alone, and
it can slip on a short phrase -- the smaller the model, the more often. If you
know what will be spoken, say so: **Spoken language** in the launcher, or
`--language fr` here. The model is then told instead of guessing.

## Setup

From the workspace root (`local_demo/`):

```bash
uv sync                    # portable engine only
uv sync --extra openvino   # also installs the OpenVINO engine
```

> **First run note:** the first time you run with a given `--model` /
> `--engine`, the model is downloaded from Hugging Face and cached locally
> (`~/.cache/huggingface`). Every run after that is fully offline.

## Usage

List available devices (microphones, output devices, and inference devices):

```bash
uv run live-translate --list-devices
```

Translate from your default microphone (portable engine, CPU):

```bash
uv run live-translate --source mic
```

Translate whatever is currently playing on the device (e.g. a video):

```bash
uv run live-translate --source system
```

Run on Intel NPU via OpenVINO:

```bash
uv run live-translate --source system --engine openvino --compute-device NPU
```

Target a specific audio device (matched by substring), pick a bigger/more
accurate model, and save the transcript:

```bash
uv run live-translate --source system --audio-device "Speakers" --model medium --output transcript.txt
```

Press `Ctrl+C` to stop.

## Subtitles, in the launcher

**Subtitles** in the launcher's panel shows the translated lines in a bar
docked at the bottom of the page: the newest line in large type, the one
before it above while both fit, dimming after a few seconds of silence. The
bar stays, and keeps receiving lines, while you open other demos -- so it is
in the picture when the launcher's window is shared in a call, in any
browser.

**Keep on top**, on the bar, moves the subtitles into a small window the
browser keeps above every other application (Document Picture-in-Picture:
Chrome and Edge), for subtitles over slides or another program. Closing that
window brings them back to the bar. A browser that cannot do it says so and
keeps the bar.

## The transcript, and a summary of it

The launcher keeps everything live translation has heard, and keeps it across
Stop and Start: stopping to change the spoken language half-way through is
still the same meeting. A reloaded page gets its lines back. **Clear
transcript** starts a new one, and so does restarting the launcher.

**Summarise in Meeting Notes** hands the whole transcript to
[meeting-notes](../meeting-notes/README.md), which writes a summary and the
action items from it, on the chip chosen in its own panel. Nothing is
transcribed a second time, live translation keeps running, and pressing the
button again later summarises everything heard up to then. The transcript is
the English translation, so the notes are in English whatever was spoken.

Over the API: `GET /api/live-translation/transcript` reads it, `DELETE` on the
same address clears it, and `POST /api/meeting-notes/from-live-translation`
hands it over (then `POST /api/meeting-notes/generate` as usual).

## Options

| Flag | Description |
| --- | --- |
| `--source {mic,system}` | Capture from a microphone or system audio loopback. Default: `mic`. |
| `--audio-device NAME` | Substring to match a specific microphone/output device name. Default: system default. |
| `--engine {portable,openvino}` | Inference backend. Default: `portable`. |
| `--model NAME` | Model size: `tiny`, `base`, `small`, `medium`, `large-v3`. Default depends on `--engine` (`small` for portable, `base` for openvino). |
| `--compute-device NAME` | Device to run on. For `portable`: `cpu`, `cuda`, `auto`. For `openvino`: `AUTO`, `CPU`, `GPU`, `NPU`. Default: `cpu` for portable, `AUTO` for openvino. |
| `--compute-type NAME` | `portable` engine only — faster-whisper compute type (`int8`, `float16`, `float32`, ...). Default: `auto`. |
| `--language CODE` | The language being spoken (`fr`, `de`, `es`, ... -- see `languages.py`). Default: detected for each utterance. |
| `--model-path PATH` | `openvino` engine only — use a model you converted yourself instead of Intel's default pre-converted one (the older `--ov-model-dir` spelling still works). |
| `--output FILE` | Also append each translated line to a text file. |
| `--list-devices` | List available microphones, output devices, and inference devices, then exit. |

## How it works

1. **Capture** ([`pantherlake_ai_core.audio`](../../core/src/pantherlake_ai_core/audio.py),
   shared with other bricks) — `soundcard` pulls small audio blocks from the
   microphone or, for system audio, opens the default output device in
   WASAPI loopback mode so it can record whatever the speakers are playing.
2. **Segment** ([`pantherlake_ai_core.segmenter`](../../core/src/pantherlake_ai_core/segmenter.py),
   shared) — a lightweight energy-based voice-activity detector groups those
   blocks into individual speech utterances, so each model call gets one
   coherent chunk of speech rather than arbitrary fixed-length slices.
3. **Translate** ([`transcriber.py`](src/live_translation/transcriber.py)) —
   a factory picks [`transcriber_portable.py`](src/live_translation/transcriber_portable.py)
   or [`transcriber_openvino.py`](src/live_translation/transcriber_openvino.py)
   based on `--engine`; both expose the same `.translate(audio)` call so the
   CLI doesn't care which one is loaded. `create_translator(..., task=...)`
   also accepts `task="transcribe"` for same-language speech-to-text instead
   of always-English output -- this CLI never sets it (translation is the
   whole point here), but [`voice-assistant`](../voice-assistant/README.md)
   reuses this same factory that way rather than writing a second Whisper
   wrapper.

## Using a custom OpenVINO model

The `openvino` engine downloads Intel's pre-converted multilingual models
(`OpenVINO/whisper-{tiny,base,medium,large-v3}-fp16-ov` on Hugging Face) by
default. To use a different size, a quantized variant, or a fine-tuned
model, convert it yourself:

```bash
uv run --extra openvino optimum-cli export openvino --trust-remote-code --model openai/whisper-small ./whisper-small-ov
uv run live-translate --engine openvino --model-path ./whisper-small-ov
```

## Tuning for your setup

- Speech is told from the room by level: anything well above the noise floor,
  which is measured continuously from the quietest moments of the last ten
  seconds. You can be talking when you press Start, and a microphone whose
  level moves during a call is followed. If speech still gets cut off or
  missed on your room/device — adjust `VADConfig` in
  [`pantherlake_ai_core.segmenter`](../../core/src/pantherlake_ai_core/segmenter.py)
  (e.g. `threshold_multiplier`, `silence_hangover`).
- If translations lag behind live audio, drop to a smaller `--model` (`tiny`
  or `base`), use `--compute-device cuda` (portable, NVIDIA GPU), or use the
  `openvino` engine with `--compute-device NPU`/`GPU` on Intel hardware.
- `--model large-v3` gives the best accuracy but is the slowest on CPU.
