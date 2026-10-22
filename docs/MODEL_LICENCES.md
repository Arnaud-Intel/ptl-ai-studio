# Model licences

Every model the app fetches, the licence its publisher puts on it, and what
has been decided about the ones that are not free to use anywhere. This is
backlog ticket **R17**.

**Read on 2026-10-12**, at v1.0.5. For a model on the Hugging Face Hub, the
licence, the gating and the revision are what the Hub's own API answered that
day for the repository the app downloads from, and again for the model it was
made from. For a model a Python package fetches by itself, they are what the
installed package says. A row is what a publisher states, not a legal opinion:
whether a licence suits a given use is the product owner's call, and the
decisions below are theirs to sign.

The list is the app's own (`core/src/pantherlake_ai_core/models.py`, where each
model now carries its `licence`); `tests/test_models_prefetch.py` fails when a
model is added without one, when this page does not name it, or when a licence
that is not Apache-2.0 or MIT has no decision here.

## In short

- **22 entries, 19 of them Apache-2.0 or MIT**, none gated, none asking for
  an account.
- **Two are not free to use anywhere**: the detector, YOLO11 (**AGPL-3.0**),
  and the wake word (**CC BY-NC-SA 4.0**, non-commercial). The wake word is
  **decided** (2026-10-13): a button to talk is the default, the wake word a
  choice that loads nothing until it is made. The detector is **to be
  replaced** (decided the same day); four Apache-2.0 detectors were tried
  and YOLOX-s is proposed, the app still loading YOLO11s until that is
  agreed: [Decisions](#decisions).
- **Two depend on somebody else's repository** without a licence problem:
  the image model, a community build held at one revision, and the portable
  detector, whose build names no licence of its own.
- **One model was missing from the list** and is in it now: WavMark, which
  OpenVoice fetches the first time a voice is cloned (MIT).
- The sample videos carry four different Creative Commons terms and are
  credited in [`sample-data/videos`](../sample-data/videos/README.md).

## What the app fetches

"Revision" is the state of the repository the app was tried with: on
2026-10-12 it was, for every row, both what is on the demo laptop and what the
Hub serves. Only the image model is *held* at its revision; the others fetch
whatever the repository holds on the day (R10).

| Key | Model | Demos | Repository @ revision | Licence | Made from | Gated |
| --- | --- | --- | --- | --- | --- | --- |
| `whisper-medium-ov` | Whisper medium (speech translation) | Live Speech Translation, Live Meeting Notes | [`OpenVINO/whisper-medium-fp16-ov`](https://huggingface.co/OpenVINO/whisper-medium-fp16-ov) @ `eddc8397f55a` | Apache-2.0 | openai/whisper-medium (Apache-2.0) | no |
| `whisper-base-ov` | Whisper base (speech) | Local Voice Assistant | [`OpenVINO/whisper-base-fp16-ov`](https://huggingface.co/OpenVINO/whisper-base-fp16-ov) @ `84fbe975a79a` | Apache-2.0 | openai/whisper-base (Apache-2.0) | no |
| `whisper-small-portable` | faster-whisper small (speech) | Live Speech Translation, Live Meeting Notes, Local Voice Assistant | [`Systran/faster-whisper-small`](https://huggingface.co/Systran/faster-whisper-small) @ `536b0662742c` | MIT | openai/whisper-small | no |
| `llm-1.5b-ov` | Qwen2.5 1.5B (text) | Local Document Q&A, Local Voice Assistant, Expense Report Extractor, Local Screen Memory, Video Commentator | [`OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov`](https://huggingface.co/OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov) @ `4d14c299e35d` | Apache-2.0 | Qwen/Qwen2.5-1.5B-Instruct (Apache-2.0) | no |
| `notes-8b-ov` | Qwen3 8B (meeting notes, page plans) | Live Meeting Notes, Page Agent | [`OpenVINO/Qwen3-8B-int4-ov`](https://huggingface.co/OpenVINO/Qwen3-8B-int4-ov) @ `5c47abf4b8e1` | Apache-2.0 | Qwen/Qwen3-8B (Apache-2.0) | no |
| `notes-8b-npu-ov` | Qwen3 8B, NPU build | Live Meeting Notes, Page Agent | [`OpenVINO/Qwen3-8B-int4-cw-ov`](https://huggingface.co/OpenVINO/Qwen3-8B-int4-cw-ov) @ `27ae5b116eb7` | Apache-2.0 | Qwen/Qwen3-8B (Apache-2.0) | no |
| `flux2-klein-ov` | FLUX.2 klein 4B (draws pictures, redraws a portrait) | Page Agent, Portrait Studio | [`rupeshs/flux2-klein-4b-int4-ov`](https://huggingface.co/rupeshs/flux2-klein-4b-int4-ov) @ `2adc62976ad3`, held | Apache-2.0 | black-forest-labs/FLUX.2-klein-4B (Apache-2.0) | no |
| `llm-1.5b-portable` | Qwen2.5 1.5B GGUF (text) | Local Document Q&A, Live Meeting Notes, Local Voice Assistant, Expense Report Extractor, Local Screen Memory | [`Qwen/Qwen2.5-1.5B-Instruct-GGUF`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF) @ `91cad51170dc` | Apache-2.0 | the publisher's own build | no |
| `embed-ov` | Qwen3 embeddings | Local Document Q&A, Local Screen Memory | [`OpenVINO/Qwen3-Embedding-0.6B-int8-ov`](https://huggingface.co/OpenVINO/Qwen3-Embedding-0.6B-int8-ov) @ `dfe67308c933` | Apache-2.0 | Qwen/Qwen3-Embedding-0.6B (Apache-2.0) | no |
| `embed-portable` | nomic embeddings GGUF | Local Document Q&A, Local Screen Memory | [`nomic-ai/nomic-embed-text-v1.5-GGUF`](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF) @ `0188c9bf4097` | Apache-2.0 | the publisher's own build | no |
| `vlm-7b-ov` | Qwen2.5-VL 7B (reads images) | Screen / Image Text Extraction, Expense Report Extractor, Local Screen Memory, Video Commentator | [`OpenVINO/Qwen2.5-VL-7B-Instruct-int4-ov`](https://huggingface.co/OpenVINO/Qwen2.5-VL-7B-Instruct-int4-ov) @ `48494b1aceaf` | Apache-2.0 | Qwen/Qwen2.5-VL-7B-Instruct (Apache-2.0) | no |
| `yolo11s-ov` | YOLO11s (detection) | Object Detection Overlay, Smart City Monitor | [`OpenVINO/YOLO11s-int8-ov`](https://huggingface.co/OpenVINO/YOLO11s-int8-ov) @ `c0b5ee29346c` | **AGPL-3.0** | Ultralytics/YOLO11 (AGPL-3.0) | no |
| `detr-portable` | DETR ResNet-50 (detection) | Object Detection Overlay, Smart City Monitor | [`Xenova/detr-resnet-50`](https://huggingface.co/Xenova/detr-resnet-50) @ `8be7ab59ff66` | none named on the build; Apache-2.0 on the original | facebook/detr-resnet-50 (Apache-2.0) | no |
| `selfie-segmentation` | Selfie segmentation (background) | Webcam Background Effects | [`onnx-community/mediapipe_selfie_segmentation`](https://huggingface.co/onnx-community/mediapipe_selfie_segmentation) @ `be49485c8e02` | Apache-2.0 | Google's MediaPipe selfie segmentation | no |
| `coder-30b-ov` | Qwen3-Coder 30B (code, HTML) | Commit & Code Review Assistant, HTML Creator, Page Agent | [`OpenVINO/Qwen3-Coder-30B-A3B-Instruct-int4-ov`](https://huggingface.co/OpenVINO/Qwen3-Coder-30B-A3B-Instruct-int4-ov) @ `21140c68a630` | Apache-2.0 | Qwen/Qwen3-Coder-30B-A3B-Instruct (Apache-2.0) | no |
| `coder-1.5b-portable` | Qwen2.5-Coder 1.5B GGUF | Commit & Code Review Assistant, HTML Creator | [`Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF`](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF) @ `f86cb2c1fa58` | Apache-2.0 | the publisher's own build | no |
| `openvoice` | OpenVoice (voice cloning, and the studio's own voice) | Voice Clone Studio | [`myshell-ai/OpenVoice`](https://huggingface.co/myshell-ai/OpenVoice) @ `c70fc8b939bd` | MIT | the publisher's own | no |
| `wavmark` | WavMark (marks a cloned voice) | Voice Clone Studio | [`M4869/WavMark`](https://huggingface.co/M4869/WavMark) @ `0ad3c7b74f64` | MIT | the publisher's own | no |
| `chatterbox` | Chatterbox Turbo (voice cloning) | Voice Clone Studio | [`ResembleAI/chatterbox-turbo-ONNX`](https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX) @ `d21799bd0354` | MIT | ResembleAI/chatterbox-turbo (MIT) | no |
| `wake-word` | Wake word "hey jarvis" | Local Voice Assistant | fetched by the `openwakeword` package (0.6.0) from its GitHub releases | **CC BY-NC-SA 4.0** | openWakeWord's own | no |
| `silero-vad` | Silero voice detection | Voice Clone Studio | `snakers4/silero-vad`, through `torch.hub`, at its `master` | MIT | the publisher's own | no |
| `street-videos` | Sample videos for counting (6 clips) | Smart City Monitor | Wikimedia Commons and GitHub, each file checked against a checksum | CC0 1.0, CC BY 3.0, CC BY 4.0, CC BY-SA 4.0 and public domain, a clip each | -- | no |

### Fetched only when asked for

| What | When | Repository | Licence |
| --- | --- | --- | --- |
| Whisper tiny, small, large-v3 on OpenVINO | another size chosen for speech | `OpenVINO/whisper-{size}-fp16-ov` | Apache-2.0, each |
| faster-whisper, other sizes | the same, on the portable engine | `Systran/faster-whisper-{size}` | MIT, each |

Nothing else is fetched by the app. Two detectors of other sizes (YOLO11n,
YOLO11m) and a speaker-verification model (`speechbrain/spkrec-ecapa-voxceleb`,
Apache-2.0) are on the demo laptop from measurements; no brick loads them.

### What comes inside a package

`openwakeword` 0.6.0 fetches four files with the wake word: `hey_jarvis_v0.1`,
its two feature models (`melspectrogram`, `embedding_model`) and a copy of
Silero's voice detection. Its README says of them: the code is Apache-2.0,
and "all of the included pre-trained models are licensed under the Creative
Commons Attribution-NonCommercial-ShareAlike 4.0 International license", for
want of knowing the terms of all the data they were trained on. It adds that
the embedding model is a re-implementation of one Google published under
Apache-2.0.

## Decisions

A model whose licence is not Apache-2.0 or MIT, or that is gated, has its
decision here: keep with its conditions, replace, or drop. Each says what is
known, what the ways out are, and what was proposed. **Both are decided; the
detector's replacement has been tried and proposed, not yet made.**

### `yolo11s-ov` -- YOLO11s, AGPL-3.0

*What it is.* The detector of Object Detection (one of the four pilot
journeys) and of the Smart City Monitor, and so of the Auto Demo's counting
scenes. Intel's card for the build says: "The original model is distributed
under GNU Affero General Public License v3.0". Every size of YOLO11 that
Intel publishes carries it.

*What it asks, as far as a reader who is not a lawyer can say.* AGPL-3.0 is a
copyleft licence: what is built on the work and passed on, or offered over a
network, has to be offered under the same licence with its source. Its
publisher, Ultralytics, sells another licence to those who do not want that.
Whether a studio that downloads the weights at run time, and whose own
repository is public but states no licence, is within the terms is the
question to put to whoever answers for licences.

*Ways out.*

1. **Replace it** with a detector under Apache-2.0. Intel publishes several
   already converted for OpenVINO, so no export is needed:
   `OpenVINO/yolox_s-int8-ov` (10 MB), `OpenVINO/deimv2_s-int8-ov` (13 MB),
   `OpenVINO/rfdetr_small-int8-ov` (31 MB), `OpenVINO/rtdetr_50-int8-ov`
   (46 MB), all Apache-2.0 on their cards, not gated. None has been run here:
   frames per second on each chip, what each finds on the sample clips, and
   whether it loads through the same `openvino-model-api` call are to measure
   -- a day or two, and R03 and R12 then measure on the one chosen. The
   portable engine's detector, DETR, is Apache-2.0 already.
2. **Keep it**, with the conditions written down by somebody entitled to:
   the app's source published under a compatible licence, or Ultralytics'
   commercial licence bought.
3. **Drop the two demos** from what is shown for a company. They are the
   plainest picture of "a chip per feed".

*What Ultralytics itself says* (its licence page and its model card, read
2026-10-13): "All Ultralytics YOLO trained models fall under the AGPL-3.0";
a project that uses them is to be open-sourced whole under AGPL-3.0, or
covered by its Enterprise License, which is the one "for development and
production use ... business products and services, including internal tools,
automated workflows, and production deployments". That is a publisher's
reading of its own licence, wider than the licence's text on running a
program; it is also the reading a partner's legal department will find.

*Proposed:* replace (1), after a trial of the four.

**Decided by the product owner on 2026-10-13: replace, after a trial** -- told
first what the licence means here, then: "download them, measure speed and
counts, and propose one". The trial is below. **The app still loads YOLO11s:
which detector takes its place is the next word to wait for.**

#### The trial (2026-10-13)

Four detectors from Intel's Hub organisation, 98 MB in all, each loaded the
way the brick loads its own (`model_api`, the app's compile settings), one
process at a time, against YOLO11s. XPS 14 on mains with the B60 docked and
idle, not held back (36 W with eight cores busy). A frame is a 1080p frame of
the Toronto clip through the whole of `detect()`: resize, model, decoding.

| | Integrated GPU | NPU | CPU | What it gives |
| --- | --- | --- | --- | --- |
| YOLO11s (today, AGPL-3.0) | 6.9 to 10.3 ms, 90 to 143 a second | 22 ms, 45 a second | 21 ms, 46 a second | the reference |
| **YOLOX-s** | 6.4 to 9.7 ms, 102 to 152 | **14 ms, 70** | 27 ms, 36 | far more small people; some cattle called horses |
| DEIMv2-s | 26 ms, 36 | 44 ms, 21 | 119 ms, 8 | a little more than YOLO11s, clean |
| RT-DETR-50 | 27 ms, 37 | does not compile ("Bad optional access") | 182 ms, 5.5 | a little more than YOLO11s, clean |
| RF-DETR-small | 24 ms, 43 | 33 ms, 29 | 84 ms, 11 | **unusable as published**: all 300 candidates score 0.8 or more, 200 boxes a frame |

Energy flat out, above idle, a frame: YOLO11s 101 to 119 mJ on the GPU, 136
to 143 on the NPU, 697 on the CPU; YOLOX-s 80 to 105, 110 to 121, 544. Held
to 30 frames a second both add 2 to 7 W to the package on the GPU or the NPU,
which is no more than idle itself moved between runs (3.3 to 6.9 W).

What each finds on the six sample clips, on the integrated GPU, every second
frame (12 to 15 a second), at the app's threshold of 0.5: boxes on screen
(median), then what the Smart City Monitor's own tracker would have counted.

| | Toronto | Tyumen | Tokyo | Intel street | Bottle line | Cattle |
| --- | --- | --- | --- | --- | --- | --- |
| YOLO11s | 11 / 469 | 9 / 207 | 3 / 238 | 0 / 18 | 3 / 15 | 9 / 130 |
| YOLOX-s | 21 / 1,121 | 16 / 449 | 36 / 2,779 | 0 / 15 | 2 / 33 | 12 / 292 |
| DEIMv2-s | 14 / 621 | 12 / 244 | 6 / 474 | 0 / 13 | 3 / 20 | 10 / 133 |
| RT-DETR-50 | 15 / 697 | 16 / 296 | 6 / 506 | 0 / 13 | 3 / 22 | 12 / 163 |

YOLOX-s finds the same on the three chips, as YOLO11s does (Toronto: 21, 22
and 21 on screen on the GPU, the NPU and the CPU; 1,121, 1,116 and 1,103
counted). On the integrated GPU the two are as fast as each other: each
moved by a third from one run to the next.

Looked at, two frames a clip, the four side by side:

- **The crowd.** In the Tokyo crossing, seen from above, YOLO11s boxes the
  cars and nobody; the two DETRs one to three people; YOLOX-s some thirty-five,
  and they are people. Across the street in Toronto it finds the row of people
  waiting, where YOLO11s finds one. This is the fault R03 has carried since
  September ("small people in a crowd are mostly missed").
- **The herd.** YOLOX-s calls some cattle horses: 2.6 "horse" boxes a frame
  where there is one horse, the rider's (YOLO11s: 0.65), and once put
  "person" on a cow's head. The others do not.
- **The bottles.** All find them. In one of the two frames looked at YOLOX-s
  missed the bottle under the capping head; over the clip it boxes as many
  as YOLO11s (2.2 a frame against 2.3).

*What YOLOX-s costs.* (1) The library decodes it slowly: `model_api` thins
out the overlapping boxes among all 8,400 candidates before dropping the weak
ones, in Python, 180 ms a frame. Dropping them first gives the same boxes
(but for ties between two boxes of equal score, a few pixels apart, in 3
frames of 30) in under a millisecond: some twenty lines in the brick, and
every figure above is with them. (2) **The counts get worse before they get
better.** The Monitor counts a track each time one is born, and a detector
that sees small, far things sees them come and go: Toronto 1,121 where
YOLO11s gives 469 and neither is right. At a threshold of 0.6 it is 838. R03
already says the remedy -- count at a line that is crossed -- and this makes
it the first thing to do after the swap, not a later one. (3) The horses.

*Proposed:* **YOLOX-s** (`OpenVINO/yolox_s-int8-ov`, Apache-2.0 on its card,
not gated, revision `66effd8d1d7c`). It is the only one of the four that
keeps today's frame rates on all three chips, it is half as fast again on the
NPU, and it sees the crowd. DEIMv2-s is the careful second: clean labels,
but 21 frames a second on the NPU and 8 on the CPU, which a feed per chip
does not survive. Not tried, and worth a look if the horses matter:
`OpenVINO/yolox_l-int8-ov`, the larger YOLOX, and `yolox_tiny` for the CPU.
The trial's script and its figures are in the session's scratch folder; the
four models are in the Hugging Face cache, used by nothing yet.

### `wake-word` -- "hey jarvis", CC BY-NC-SA 4.0

*What it is.* Until 2026-10-13 the Voice Assistant listened for its wake
word with one of openWakeWord's own models. They are non-commercial (above),
and so are the feature models that come with it by that README's wording.
The brick's README said "Apache-2.0 / MIT-licensed components" until
2026-10-12: that is true of the code, not of the models, and has been
corrected.

*Ways out.*

1. **A button, or a key**, in place of the wake word: "hold to talk". No
   model, works in a loud hall where a wake word does not, and half a day.
   The wake word can stay as a choice that is off unless somebody turns it
   on for a use that is not commercial.
2. **The phrase heard by Whisper**: voice detection (Silero, MIT) cuts what
   is said into short pieces and the Whisper already loaded reads them,
   waiting for the phrase. No new model and no new licence; it costs the NPU
   more than a wake-word model does, which matters to R25, and is to measure.
3. **A wake word trained for the project** with openWakeWord's code
   (Apache-2.0) on speech the project has the right to use. It would still
   sit on openWakeWord's feature models, whose licence is the same question.
4. **Ask openWakeWord's author** for other terms.

*Proposed:* (1) now, (2) as the spike that could bring a spoken wake phrase
back.

**Decided by the product owner on 2026-10-13: (1), a button to talk.** Built
the same day. The Voice Assistant starts with no wake word: a question is
what is said while the panel's **Hold to talk** button, or the space bar, is
held (Enter, from a terminal). The wake words are still in the menu, each
marked "non-commercial model", for a use that is not commercial; nothing of
openWakeWord's is loaded or fetched until one is chosen, and the "Start
small" set of the setup assistant no longer fetches it. The `openwakeword`
package itself (Apache-2.0) is still installed. (2) was not asked for.

### Not a licence problem, to know about

- **`flux2-klein-ov`.** Apache-2.0 on the build and on Black Forest Labs'
  original, not gated. But Intel publishes no OpenVINO build, so the app
  fetches one person's (`rupeshs/flux2-klein-4b-int4-ov`), held at revision
  `2adc62976ad3`. If that repository went away, a new install could not draw:
  the Page Agent, Portrait Studio and the Auto Demo's first scene. An export
  of our own, or the decision to stay, is ticket **R52**.
- **`detr-portable`.** The build's page (`Xenova/detr-resnet-50`) names no
  licence; the model it converts, `facebook/detr-resnet-50`, is Apache-2.0.
  Proposed: keep, on the original's licence.
- **`silero-vad`** is fetched at the `master` of its repository, not at a
  revision: MIT today, and whatever that branch holds tomorrow (R10).
- **FLUX.1-schnell** drew the four bundled portraits and left the app on
  2026-10-12. Apache-2.0; nothing about its pictures is restricted by it.

## Models measured and not in the app

Read the same day, so that adopting one is not a second review. "Build" is
what would be fetched; "original" is what it was made from.

| Build | Licence, build / original | Gated | For |
| --- | --- | --- | --- |
| [`HarmenWessels/K2-Horizon-7B-int4-symg128-ov`](https://huggingface.co/HarmenWessels/K2-Horizon-7B-int4-symg128-ov), and `-3.7B-` | Apache-2.0 / Apache-2.0 (`IFM/K2-Horizon-7B`) | no | R36. One person's build, as the image model is: to hold at a revision, or to export |
| [`OpenVINO/gemma-4-E2B-it-int4-ov`](https://huggingface.co/OpenVINO/gemma-4-E2B-it-int4-ov), `-E4B-`, `-26b-a4b-` | Apache-2.0 / Apache-2.0 (`google/gemma-4-*`) | no | R36, R37. On the NPU only once re-compressed, which makes a build that exists on one laptop |
| [`OpenVINO/Qwen3.5-4B-int4-ov`](https://huggingface.co/OpenVINO/Qwen3.5-4B-int4-ov), `Qwen3.6-35B-A3B`, `Qwen3.8-27B`, `Qwen3-VL-4B-Instruct` | Apache-2.0 / Apache-2.0 | no | R37, R24, R23 |
| [`OpenVINO/Qwen3-Embedding-0.6B-int4-cw-ov`](https://huggingface.co/OpenVINO/Qwen3-Embedding-0.6B-int4-cw-ov) | Apache-2.0 / Apache-2.0 | no | R37 |
| [`OpenVINO/whisper-large-v3-turbo-fp16-ov`](https://huggingface.co/OpenVINO/whisper-large-v3-turbo-fp16-ov) | MIT / MIT | no | R37 |
| [`OpenVINO/LCM_Dreamshaper_v7-int8-ov`](https://huggingface.co/OpenVINO/LCM_Dreamshaper_v7-int8-ov) | MIT / MIT (`SimianLuo/LCM_Dreamshaper_v7`) | no | R37. It distils `Lykon/dreamshaper-7`, which is CreativeML OpenRAIL-M, a licence with rules on use: to read before it is adopted |
| [`OpenVINO/gpt-oss-20b-int4-ov`](https://huggingface.co/OpenVINO/gpt-oss-20b-int4-ov), `Qwen3-30B-A3B-Instruct-2507` | Apache-2.0 / Apache-2.0 | no | R26 |
| [`OpenVINO/Kokoro-82M-int8-ov`](https://huggingface.co/OpenVINO/Kokoro-82M-int8-ov) | Apache-2.0 / Apache-2.0 | no | measured and set aside (R37) |
| [`OpenVINO/gemma-3-4b-it-int4-cw-ov`](https://huggingface.co/OpenVINO/gemma-3-4b-it-int4-cw-ov) | Gemma terms | **yes**: terms to accept with an account | not tried, for that reason |

Not read here, because nothing was measured: the anomaly models and practice
pictures of R42 (MVTec's set is non-commercial), a face detector for R44, a
pose model for R49 (YOLO's is AGPL-3.0 like the detector), an image-text
model for R29.

## Seen on the way, which is not a model

Filed in the backlog's Inbox on 2026-10-12, for R16's review of what goes out
with a release:

- **The repository states no licence of its own.** It is public; without one
  nobody outside has been given the right to use or change the code, and the
  AGPL question above cannot be answered either way.
- **Two installed packages are copyleft**: `mutagen` (GPL-2.0-or-later),
  asked for by `openwakeword` and `yt-dlp`, and `soxr` (LGPL-2.1-or-later),
  asked for by `librosa`. `uv` installs them on the user's machine; the
  project does not ship them.
- **The live city cameras** are other people's streams: YouTube's, read with
  `yt-dlp`, and Transport for London's. Their terms were not read for this
  ticket. Since 2026-10-09 the demos open on the videos kept on the machine.

## Keeping this page true

When a model is added to `models.py`, give it its `licence` and, if the build
is not the original's, its `made_from`; add its row here; and if the licence
is not in `models.PERMISSIVE`, write its decision before the brick is shown.
The Hub's answer for a repository is one call:

```bash
uv run python -c "from huggingface_hub import HfApi; i = HfApi().model_info('OpenVINO/YOLO11s-int8-ov'); print(i.sha, i.gated, i.card_data.license)"
```
