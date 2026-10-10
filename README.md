<p align="center">
  <img src="docs/panther-lake-ai-studio-banner.png" alt="Panther Lake AI Studio" width="720" />
</p>

<p align="center">
  <a href="https://github.com/Arnaud-Intel/ptl-ai-studio/tags"><img src="https://img.shields.io/github/v/tag/Arnaud-Intel/ptl-ai-studio?label=version&color=0068B5" alt="Version" /></a>
  <img src="https://img.shields.io/badge/Intel%20Core%20Ultra-Lunar%20Lake%20%C2%B7%20Panther%20Lake-0068B5" alt="Intel Core Ultra: Lunar Lake and Panther Lake" />
  <a href="https://docs.openvino.ai/"><img src="https://img.shields.io/badge/runtime-OpenVINO-8A2BE2" alt="OpenVINO" /></a>
  <img src="https://img.shields.io/badge/cloud%20calls-zero-4ade80" alt="Cloud calls: zero" />
  <img src="https://img.shields.io/badge/platform-Windows%2011-0078D6" alt="Platform: Windows 11" />
  <a href="https://github.com/Arnaud-Intel/ptl-ai-studio/actions/workflows/test.yml"><img src="https://github.com/Arnaud-Intel/ptl-ai-studio/actions/workflows/test.yml/badge.svg" alt="Tests" /></a>
</p>

**Fifteen AI demos that run on the laptop in front of you, and a show that
plays them by itself. For Intel Core Ultra laptops: Lunar Lake and Panther
Lake. One launcher, no cloud.**

- **Every demo runs on the machine**: no API key, no account, nothing sent
  anywhere. Once its models are downloaded it works with the network off.
- **You choose the chip** -- CPU, integrated GPU or NPU -- and a hardware
  panel shows which one is working, how fast, and how many watts it costs.
- **Auto Demo** plays the demos in a loop and tells each one as a short
  story, in English or French: for a stand, a meeting, or a screen nobody
  is driving.

<p align="center">
  <img src="docs/screenshot-stage-page.png" alt="The Auto Demo's stage: a caption on top, the plan, six generated pictures and the finished web page in the middle, and the NPU, integrated GPU and CPU down the right with what each one did" width="860" />
</p>
<p align="center"><sub>One request, one web page: planned on the NPU, illustrated and written on the integrated GPU, in under four minutes, with no graphics card.</sub></p>

## Which laptops

| | Lunar Lake | Panther Lake |
| --- | --- | --- |
| Processor | Intel Core Ultra 200V series | Intel Core Ultra series 3 |
| Chips the demos use | CPU, Arc integrated GPU, NPU | CPU, Arc integrated GPU, NPU |
| System | Windows 11, current Intel graphics and NPU drivers | the same |

- **No graphics card is needed.** A discrete GPU is used when there is one
  (the Page Agent then runs its two GPU jobs at once) and never required.
- **Memory decides which demos fit.** The models live in system memory,
  which the integrated GPU shares. Most demos need a few gigabytes; the
  three that use the 30B coding model (HTML Creator, Code Review, Page
  Agent) need about 17 GB for it, and say so on their cards. Code Review
  and HTML Creator can run a 1.5B model on the CPU instead.
- **What was measured where.** Every figure and screenshot on this page
  comes from a Panther Lake laptop (Dell XPS 14, Core Ultra X7 358H, 64 GB).
  A Lunar Lake laptop runs the same build on the same three chips, at its
  own speeds.
- **Another PC?** The demos fall back to their CPU engines, and the options
  that need an Intel GPU or NPU are greyed out with the reason.

## Install on a new device

You need Windows 11, an internet connection for the installation, and
disk space for the models you want: 150 MB for a first speech demo, about
45 GB for all of them. No administrator rights.

1. **Update the Intel drivers** (graphics and NPU), for instance with the
   [Intel Driver & Support Assistant](https://www.intel.com/content/www/us/en/support/detect.html).
   The installer does not touch drivers, and the NPU demos need a recent one.
2. **Get the project.** With git (this is what lets the app upgrade itself
   later):
   ```bash
   git clone https://github.com/Arnaud-Intel/ptl-ai-studio.git
   ```
   Or **Code > Download ZIP** on this page, and extract all of it.
3. **Double-click `first_launch.bat`** in the project folder. It walks
   through five steps and asks before each download:
   1. finds [`uv`](https://docs.astral.sh/uv/), or offers to install it (uv brings its own Python);
   2. installs the environment with the OpenVINO engines (a few minutes);
   3. lists the chips it found -- CPU, GPU, NPU -- and the free disk space;
   4. fetches models: a speech starter, one demo's, or all of them, sizes shown first, then the sample videos (113 MB);
   5. starts the Studio and opens `http://127.0.0.1:8765`.
4. **From then on, double-click `start_launcher.bat`.** Close its window
   to stop the Studio, or run `stop_launcher.bat`.

The helper can be run again: what is already downloaded is kept. Its log
is in `logs/first-launch-*.log`. On a company network, a proxy or
certificate error is for your IT team to settle; the helper changes no
security setting.

<details>
<summary><b>Installing by hand, offline use, updates</b></summary>

```bash
git clone https://github.com/Arnaud-Intel/ptl-ai-studio.git
cd ptl-ai-studio
uv sync --extra openvino      # without --extra: CPU engines only
uv run panther-lake-launcher
```

- **Models** are fetched the first time a demo needs them. To fetch them
  ahead, use **Prepare models** in the footer (what each is for, its size,
  whether this machine has it), or `uv run panther-lake-prefetch`
  (`--list` to look, `--demo doc-qa` for one demo).
- **Offline.** Install and fetch the models first; after that the Studio
  starts and runs with the network off. Only live city cameras and the
  update check want a connection.
- **Updates.** A copy made with git checks GitHub when it starts and offers
  the new version with what changed; **Upgrade** in the footer does the
  rest and restarts. A copy from a ZIP is updated by downloading it again.
- **YouTube city cameras** in the Smart City demo also need Deno or Node;
  the helper checks and offers to install Deno. Everything else works
  without. See the [Smart City notes](bricks/smart-city-monitor/README.md#connection-recovery-and-setup-checks).

</details>

## Two ways in

<p align="center">
  <img src="docs/screenshot-start.png" alt="The start screen: Auto Demo or Manual demo" width="49%" />
  <img src="docs/screenshot-autodemo-start.png" alt="Starting the Auto Demo: what this machine has, the language, the camera, and a tick for each scene" width="49%" />
</p>

### Auto Demo

The Studio presents itself: one demo on screen at a time, a caption that
says what is happening and on which chip, and the chips at work down the
right. Before it starts you choose the language, which scenes play, and
whether the camera may be used. A click brings up *keep playing*, *pause*
or *stop*; left alone, it carries on. It makes no sound.

| Scene | What it shows | Chips |
| --- | --- | --- |
| Three models, three chips, one web page | A request becomes an illustrated page: planned, drawn, written | NPU + GPU |
| The camera sees you | Whoever is in front is boxed live, and a vision model says what is going on. Nothing is recorded | NPU + GPU |
| Two chips share one job | Five worn receipts become expense lines; a doubtful figure is flagged, not trusted | GPU + NPU |
| Street cameras / Count whatever passes | Two videos counted at once, one per chip: streets one turn, a herd and a bottling line the next | GPU + NPU |
| The same question, without the files and with them | Asked of the model alone it makes something up; asked again with the files, it answers from them | NPU |
| A video, watched and commented on | A plain sentence about each frame, then said again as a sports commentator or a nature documentary | GPU + NPU |

<p align="center">
  <img src="docs/screenshot-stage-receipts.png" alt="Five receipts, each beside the expense line made of it; one amount is flagged to check" width="49%" />
  <img src="docs/screenshot-stage-streets.png" alt="Two street videos with boxes drawn on people and cars, one on the integrated GPU and one on the NPU, both at 24 frames a second" width="49%" />
</p>
<p align="center">
  <img src="docs/screenshot-stage-documents.png" alt="One question asked twice: alone the model invents an answer, with the files it gives the right one, and the files it used are marked" width="49%" />
  <img src="docs/screenshot-stage-commentary.png" alt="A video of cattle on a road with a spoken-style comment as a subtitle, and what the vision model actually saw underneath" width="49%" />
</p>

How it is built, and what was measured: [docs/AUTO_DEMO.md](docs/AUTO_DEMO.md).

### Manual demo

Every demo with its settings. Open one, pick an engine and a chip, press
Start. Leaving a demo does not stop it: the hardware panel keeps every
running demo in view under the chip it is on, with its own figure (frames
or tokens per second) and a button to stop it. Demos that need content
come with samples: a fictional company's documents, receipts, street
videos, prompts.

<p align="center">
  <img src="docs/screenshot-demo.png" alt="The Expense Report Extractor at work: receipts read on the integrated GPU while the NPU fills in the lines, both shown in the hardware panel with their speed and the power drawn" width="860" />
</p>

## The demos

| Demo | What it does | Chips |
| --- | --- | --- |
| **Live Speech Translation** | Any spoken language to English text, live | CPU · GPU · NPU |
| **Live Meeting Notes** | Transcribes a call; summary and action items on demand | CPU · GPU · NPU |
| **Local Voice Assistant** | Wake word, question, spoken answer | CPU · GPU |
| **Voice Clone Studio** | Enrol a short voice sample, then speak any text in that voice | CPU |
| **Webcam Background Effects** | Background blur or replacement, live | CPU · NPU |
| **Object Detection Overlay** | Labelled boxes on a video file, a webcam or the screen | CPU · GPU · NPU |
| **Screen / Image Text Extraction** | Reads the text in a screenshot or a photo, and can translate it | CPU · GPU |
| **Smart City Monitor** | Counts people, cars and bikes on videos or live cameras | one chip per feed |
| **Video Commentator** *(experimental)* | Watches a video and says what is happening, in a mood, aloud if you like | GPU + NPU |
| **Local Document Q&A** | Answers from your own files and names them; can be asked without them, to compare | CPU · GPU · NPU |
| **Expense Report Extractor** | A folder of receipts to reviewed expense lines and an Excel file | two chips at once |
| **Local Screen Memory** | Indexes what was on screen so it can be searched later | two chips at once |
| **Commit & Code Review Assistant** | A git diff to a commit message and review notes | CPU · GPU |
| **HTML Creator** | A description, or a folder of documents, to one self-contained web page | CPU · GPU |
| **Page Agent** *(experimental)* | One request to an illustrated page: three models, conducted by plain code | NPU + GPU |

Each demo is a "brick" with its own README under [`bricks/`](bricks), and
its own command:

```bash
uv run doc-qa ./my-notes --engine openvino --compute-device NPU
uv run object-detect --source file --engine openvino --compute-device GPU
uv run expense-extract ./receipts --ocr-engine openvino --ocr-device GPU --llm-engine openvino --llm-device NPU
```

`--list-devices` shows what the machine can target; `--help` has the rest.

## Good to know

- **The numbers are real.** A chip's row shows the device the demo handed
  to OpenVINO, not a guess. Power is the processor's own energy counter,
  and an answer says what it cost over idle.
- **It says what it cannot do.** A model that does not compile for a chip
  is greyed out with the reason; a figure the receipts demo cannot find on
  the receipt is flagged; the commentator shows what it saw beside what it
  said.
- **A first run is slow.** A demo downloads its model, then compiles it for
  the chip; the status line says which. The Activity Log (top right) has
  the history, errors included.
- **Known limit.** The Voice Assistant and the Voice Clone Studio's
  OpenVoice model must not be given the NPU for now: their voice model
  does not compile there and takes the Studio down with it. Use the CPU.
- **Port taken?** A Studio is already running: `stop_launcher.bat`, or
  start another with `uv run panther-lake-launcher --port 8766`.

## For contributors

One [`uv` workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/):
a `core` package, one package per brick, and a FastAPI launcher with a
plain HTML/CSS/JS front end, no build step. Bricks reuse each other rather
than copy: Meeting Notes has no model of its own, the Smart City Monitor
borrows the detector, the Commentator borrows the vision model.
[CONTRIBUTING.md](CONTRIBUTING.md) has the layout, how to add a brick and
the tests (`uv run pytest`, run on every push); [BACKLOG.md](BACKLOG.md)
has what is planned and what is known to be wrong.

The sample videos are openly licensed and credited in
[`sample-data/videos`](sample-data/videos/README.md); the sample documents
and receipts are fictional.

---

Built for [Dell](https://www.dell.com/) laptops with
[Intel(R) Core(TM) Ultra](https://www.intel.com/) processors and
[Intel(R) Arc(TM)](https://www.intel.com/) graphics.
