# voice-clone-studio

Enroll a short voice sample, then synthesize any typed text in that voice --
fully on-device. This is zero-shot voice cloning, not a training run: no
gradient descent, no wait -- enrollment is one inference pass over the
reference clip you just recorded, and every synthesis after that is another
inference pass, not a fine-tune.

Like `webcam-effects`, both engines here run the **exact same model** --
[OpenVoice](https://github.com/myshell-ai/OpenVoice) (MIT License), whose
two stages are:

- **`BaseSpeakerTTS`** -- a VITS-family model that turns text into speech in
  one of nine base delivery styles (default, whispering, shouting, excited,
  cheerful, terrified, angry, sad, friendly).
- **`ToneColorConverter`** -- takes that speech and re-colors its tone to
  match a target speaker embedding extracted from your reference clip,
  leaving the words and delivery untouched.

- **`portable`** (default) -- plain PyTorch, CPU only.
- **`openvino`** -- the identical checkpoints, converted to OpenVINO IR,
  targeting `CPU`, `GPU` (iGPU), or `NPU` explicitly.

Following Intel's own [OpenVoice -> OpenVINO conversion notebook](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/openvoice/openvoice.ipynb)
as the reference path (see **A real technical finding** below for where it
needed a fix) confirmed both engines produce near-identical output from the
same checkpoints -- switching engines here is purely a CPU-vs-NPU/iGPU
latency comparison, the same story `webcam-effects` tells for segmentation.

## Setup

From the workspace root (`local_demo/`):

```bash
uv sync                    # portable engine only
uv sync --extra openvino   # also installs the OpenVINO engine
```

This brick is this workspace's first to depend on PyTorch. That's not a
casual choice -- see **Why PyTorch is unavoidable here** below.

> **First run note:** the first time you run either engine, checkpoints are
> downloaded from Hugging Face and cached (`~/.cache/huggingface`), and a
> Silero VAD model is fetched via `torch.hub` (`~/.cache/torch/hub`). Every
> run after that is fully offline. The `openvino` engine additionally
> converts both models to OpenVINO IR once and caches the result under
> `~/.cache/panther-lake-ai-studio/voice-clone-studio/openvino_ir/` --
> that conversion takes several seconds; every run after the first reuses
> the cached IR directly.

## Usage

```bash
uv run voice-clone-studio --list-devices
```

Enroll from a file and speak a sentence in that voice:

```bash
uv run voice-clone-studio --reference my_voice.wav --text "Hello from my own cloned voice." --output cloned.wav
```

Record the reference clip from the default microphone instead of using a
file:

```bash
uv run voice-clone-studio --record 15 --text "This is what I sound like." --output cloned.wav
```

Run both stages on the Intel NPU via OpenVINO:

```bash
uv run voice-clone-studio --reference my_voice.wav --text "Hello from the NPU." --engine openvino --compute-device NPU --output cloned.wav
```

## Options

| Flag | Description |
| --- | --- |
| `--reference PATH` | Audio file of the voice to clone (5-30s of clear speech). Mutually exclusive with `--record`. |
| `--record SECONDS` | Record that many seconds from the default microphone instead of using a file. |
| `--text TEXT` | Text to speak in the cloned voice. If omitted, only enrolls and exits. |
| `--style {default,whispering,shouting,excited,cheerful,terrified,angry,sad,friendly}` | Base delivery style before tone cloning. Default: `default`. |
| `--tau FLOAT` | Tone-conversion strength -- higher tracks the reference tone more closely. Default: `0.3`. |
| `--output PATH` | Output WAV path. Default: `cloned.wav`. |
| `--engine {portable,openvino}` | Inference backend. Default: `portable`. |
| `--compute-device NAME` | `openvino` engine only: `AUTO`, `CPU`, `GPU`, `NPU`. |
| `--model-path PATH` | Reserved for a future local-checkpoint override (currently unused -- checkpoints always come from Hugging Face). |
| `--list-devices` | List microphones and inference devices, then exit. |

## How it works

1. **Enroll** ([`voice_model.enroll`](src/voice_clone_studio/voice_model.py)) --
   the reference clip is split into non-silent segments with Silero VAD
   ([`_openvoice/se_extractor.py`](src/voice_clone_studio/_openvoice/se_extractor.py)),
   and `ToneColorConverter.extract_se` averages a 256-dim tone embedding
   across them.
2. **Synthesize** ([`voice_model.synthesize`](src/voice_clone_studio/voice_model.py)) --
   `BaseSpeakerTTS` generates speech in the chosen style using its own
   fixed default-voice embedding, then `ToneColorConverter.convert`
   re-colors that audio's tone to the enrolled embedding. Both steps are
   inference; text and delivery style are decided entirely by the first
   step, voice identity entirely by the second.
3. **Engine selection** ([`cloner_portable.py`](src/voice_clone_studio/cloner_portable.py),
   [`cloner_openvino.py`](src/voice_clone_studio/cloner_openvino.py)) --
   both classes load the identical checkpoints via `voice_model.load_models()`;
   only the `openvino` engine additionally traces each model's real entry
   point (`infer`/`voice_conversion`, wrapped as `forward` by
   `voice_model.OVWrapTTS`/`OVWrapConverter`, mirroring Intel's notebook),
   converts it to OpenVINO IR, and monkey-patches the loaded PyTorch
   objects' methods to call the compiled IR instead.
4. **Session** ([`pipeline.py`](src/voice_clone_studio/pipeline.py)) --
   `VoiceCloneSession` holds one loaded cloner; enroll once, call
   `.synthesize()` as many times as you like without re-enrolling.

`voice_model.py` also exposes `load_tts_only()` and
`accelerate_tts_with_openvino()` -- the `BaseSpeakerTTS` half of the above
with no `ToneColorConverter` loaded at all, for a consumer that only needs
speech in the base voice and no cloning.
[`voice-assistant`](../voice-assistant/README.md) uses exactly this for
its spoken replies, rather than loading (and never using) a tone converter.

`src/voice_clone_studio/_openvoice/` is a trimmed vendor copy of
`myshell-ai/OpenVoice`'s core package (MIT License) -- see the next section
for why it's vendored rather than installed from PyPI/GitHub directly.

## A real technical finding: OpenVoice isn't cleanly pip-installable here

`myshell-ai/OpenVoice` has no PyPI package covering both its TTS and
tone-conversion stages (only a community fork covering tone-conversion
alone exists on PyPI). Installing straight from its GitHub `setup.py`
pulls in **hard-pinned 2023-era versions** -- `numpy==1.22.0`,
`gradio==3.48.0`, `faster-whisper==0.9.0`, `whisper-timestamped==1.14.2`,
Chinese-language NLP libraries (`pypinyin`, `cn2an`, `jieba`) -- none of
which this brick needs, and the `numpy==1.22.0` pin alone would conflict
hard with the rest of this **shared-lockfile** workspace (other bricks
need newer numpy for their own dependencies).

So `_openvoice/` vendors only the ~13 files actually needed for English
zero-shot cloning (`api.py`, `models.py`, the VITS building blocks, and an
English-only `text/` cleaner), trimmed and re-pinned to current dependency
versions instead. Three real fixes came out of that trimming, not just
mechanical copying:

- **`se_extractor.py`'s VAD path was broken by a torchaudio version
  change.** Upstream's `read_audio` helper calls `torchaudio.load`, which
  (paired with the torch/torchaudio versions this brick resolves to) now
  requires the separate `torchcodec` package to actually decode anything --
  an extra dependency with no other purpose here. Fixed by loading the
  reference clip with `librosa.load` instead (already a dependency for
  spectrogram extraction elsewhere in this package) and handing
  Silero VAD a plain tensor directly.
- **`unidecode` swapped for `anyascii`** in `text/english.py`, the same
  patch [Intel's own conversion notebook](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/openvoice/openvoice.ipynb)
  applies -- a drop-in, more permissively licensed replacement for the same
  ASCII-transliteration call.
- **OpenVINO needs a concrete `torch.nn.Module.forward`.** Neither
  `BaseSpeakerTTS` nor `ToneColorConverter` exposes one -- their real entry
  points are custom `infer`/`voice_conversion` methods. `voice_model.py`'s
  `OVWrapTTS`/`OVWrapConverter` wrap those methods as `forward`, the same
  shape the notebook uses, so `ov.convert_model` has something it can
  actually trace.

## Why PyTorch is unavoidable here

Every other brick in this workspace deliberately avoids PyTorch (ONNX
Runtime, `llama-cpp-python`, or OpenVINO's own runtime instead) to stay
fast to install. This brick can't: OpenVoice's phonemization and
spectrogram pre/post-processing stay in plain PyTorch even in Intel's own
OpenVINO conversion -- only the two heaviest submodules get offloaded to
OpenVINO IR, not the glue around them. Accepting PyTorch as a real
dependency (a CPU-only wheel, via PyTorch's own index -- see
`pyproject.toml`) was the honest option, rather than reimplementing VITS
pre/post-processing from scratch to dodge one dependency.

## Notes / current limitations

- English only in this version -- OpenVoice also ships Chinese base
  speakers, deliberately left out along with the NLP dependency chain
  (`pypinyin`, `cn2an`, `jieba`) they need; see the trimming rationale
  above.
- Very short synthesized clips can't fit the audio watermark OpenVoice
  embeds by default (`ToneColorConverter.add_watermark` needs ~0.7s per
  repeat of the embedded message) -- you'll see "Audio too short, fail to
  add watermark" printed and the clip returned unwatermarked rather than
  truncated or rejected. Longer sentences watermark normally.
- No streaming synthesis -- `.synthesize()` returns a complete clip; there
  is no token-by-token or chunked audio output.
- The launcher's "record" enrollment path captures from the machine's own
  microphone server-side (via `pantherlake_ai_core.audio`), the same way
  every other capture in this launcher works -- not through the browser's
  own microphone APIs, since the browser here is a control surface for the
  local machine's hardware, not a remote client.
