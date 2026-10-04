# Auto Demo: groundwork

Status: design, not built. Written 2026-10-04. It builds on backlog ticket
**R21** (one-click stage scenarios), which it needs first.

An Auto Demo is the app running itself on a stand with nobody at the
keyboard: it goes through a playlist of scenes, starts the demos each scene
needs -- several at once where that is the point -- says on screen what is
happening and on which chip, stops them cleanly, and starts again. For
hours.

## What a passer-by sees

A scene every one to three minutes. For each: a title, one sentence on what
to notice, the demo itself doing real work, the hardware panel showing which
chips are busy, and live numbers in the caption (tokens per second, frames
per second, watts). Touch the mouse and the loop pauses so the demo can be
tried by hand; leave it alone and the loop resumes.

## What was measured

On the XPS 14, plugged in, through the launcher's own routes -- the way an
Auto Demo would drive it.

**Several demos at once.** Object detection is on the screen as its source.

| Running | Code review | Object detection | Document Q&A, NPU | Package |
| --- | --- | --- | --- | --- |
| Code review alone, 30B on the iGPU | 45.7 tok/s | | | 44.9 W |
| Detection alone, iGPU | | 12.0 fps | | 28.8 W |
| Both on the iGPU | 43.5 tok/s | 10.3 fps | | 46.0 W |
| Q&A alone, NPU | | | 53.7 tok/s | 27.7 W |
| Detection on the iGPU, Q&A on the NPU | | 10.9 fps | 54.4 tok/s | 34.9 W |
| Three chips: detection on the iGPU, Q&A on the NPU, code review by a 1.5B model on the CPU | 33.8 tok/s (43.0 before the Q&A joined) | 7.7 fps | 36.9 tok/s | 34.1 W |

- **The integrated GPU carries a vision stream and the 30B model together**
  with little loss: 5% off the model, 14% off the frame rate.
- **The NPU does not notice GPU work**: 54 tok/s with or without it.
- **The CPU is the shared resource.** A model that takes every core
  (llama.cpp does) cost the NPU a third of its speed and the detector a
  third of its frames, because both need the CPU to feed them. A scene that
  wants "every chip at once" has to give the CPU a job that leaves cores
  free, or cap its threads.

**The same playlist eight times over** -- detection for a few seconds, a
question on the NPU, a code review on the 30B model, every model loaded and
unloaded each time: 36 to 53 s a cycle, no failure, nothing left running or
loaded after any cycle. The launcher's memory went from 0.7 GB to between
2.0 and 2.6 GB and stayed in that band without climbing. Eight cycles is
five minutes, not a day: it says the stop and unload routes are sound, not
that a ten-hour run is.

Two things the loop showed that a playlist has to allow for: the detector
needs more than six seconds from Start before it reports a frame rate, and
the NPU model answered at 30-35 tok/s when asked straight after loading
(short answers), against 54 once it had been running. Small models are
better left loaded for the whole loop.

## What can run with nobody there

| Demo | Unattended? | Needs |
| --- | --- | --- |
| HTML Creator, code review, document Q&A, screen OCR, expense extraction | Yes | Their bundled samples; nothing else |
| Object detection | Yes | A camera pointed at the aisle, or the screen as its source |
| Webcam effects | Yes | A camera; visitors see themselves |
| Smart city monitor | Only with footage | Its bundled sources are live feeds from the internet; a stand needs a clip on the laptop, which the repository does not ship |
| Live translation, meeting notes, voice assistant | Not as they are | They listen to a microphone. A stand is noisy and nobody is speaking to them; they would need a recorded clip as a source |
| Voice clone studio | No | It needs someone to enrol a voice |
| Screen memory | No | It records the screen, which on a stand is the demo itself |

So the first playlist is the language and document demos, plus vision where
there is a camera.

## How to build it

### 1. Scenes (this is R21)

A scene is data: which demos to start and with what, how long to run or
what to wait for, what to say.

```python
Scene(
    id="back-office",
    title="Two chips, one job",
    notice="The integrated GPU reads each receipt while the NPU turns the text into an expense line.",
    start=[Run("expense-extract", sample="Start here", ocr_device="GPU", llm_device="NPU")],
    until=Finished("expense-extract"), at_most=180,
    figures=["expense-extract:ocr", "expense-extract:llm", "watts"],
)
```

One click on the home page starts a scene; that is R21 as written. The Auto
Demo is a list of them played in order.

### 2. The director

Who runs the playlist?

| | The launcher | The page |
| --- | --- | --- |
| How | A thread that starts and stops demos through the runners, as the routes do | Script in the browser that fills in samples and presses the buttons |
| For | Keeps going if the page reloads or the browser stalls; can guarantee nothing is left running; knows about failures; can be tested without a browser | Shows exactly what a person would do; nothing new to draw results |
| Against | A demo that answers one request (a page, a review) returns its result to whoever asked, and here that is the launcher: the page needs to be handed the result to draw it | The show stops with the tab; no test can cover it; it breaks when the page changes |

**Proposed: the launcher directs, the page follows.** A director thread owns
the playlist, the clock and the failure rules, and publishes its state at
`GET /api/autodemo`: scene, step, caption, and for request-style demos the
result it received. The page, in follow mode, opens the scene's panel,
draws that result with the panel's own drawing code, and shows the text as
it is written through the existing partial-text route. A reload picks up
mid-scene.

### 3. Commentary

- **Captions, written per scene, with live figures filled in.** "The
  integrated GPU is writing this page at {html-creator} tokens a second,
  from a 30B model, in {watts} W." The figures come from the numbers the
  hardware panel already shows. Reliable, and the claims are ours.
- **Spoken, optional and off by default.** The voice assistant already has
  a synthetic voice (no real person's) and the launcher can play audio, so
  a caption can be read out. On a stand, sound is often unwelcome.
- **Written by a model, later if wanted.** A model could turn the figures
  into a fresh sentence each loop. The small NPU model is not good enough
  to be trusted with claims in front of an audience, and the large one is
  busy; scripted text with real numbers says more.

This is not R23 (a model describing a video feed), though a narration scene
would slot into the playlist once R23 exists.

### 4. Running for hours

| Hazard | Handling |
| --- | --- |
| Windows sleeps, dims or locks | Hold a keep-awake request while the loop runs (nothing in the app does today); a lock forced by company policy cannot be overridden and has to be lifted for the stand |
| A scene fails | Log it, skip it, carry on; after three failures in a row, stop and show a plain notice rather than loop on an error |
| The update prompt, the models dialog | No dialogs in follow mode; never upgrade during a loop |
| A model or a camera is missing | Checked before the loop starts (the models list already knows); camera scenes are skipped without one |
| No network | The first playlist needs none |
| On battery | The integrated GPU ran a third slower on battery (backlog Inbox); warn before starting |
| Heat | The 30B model went from 38 to 30 tok/s after many runs in a row; alternate heavy and light scenes, with a title card between |
| Memory | Stop and unload were clean over eight cycles; keep the small models loaded, load the 30B and the 7B per scene |
| Things pile up | Each expense run saves a report: the loop needs to delete its own report, which needs a delete-one route (today it is all or nothing). The activity log already rotates |
| The same page twice | The HTML Creator scenes run with "Same page every time", so the audience gets the pages that were checked |
| A visitor wants a go | Mouse, key or touch pauses the loop and says so; it resumes after two idle minutes |
| People on camera | Nothing is recorded; a line on screen says so |

### 5. A first playlist

| Scene | Demos | Time | What it shows |
| --- | --- | --- | --- |
| Two chips, one job | Expense extraction: OCR on the iGPU, structuring on the NPU | ~1 min | Both chips lit at once |
| A page from a sentence | HTML Creator, one of the five scenarios, then full screen | 2-3.5 min | A 30B model with no discrete GPU; text as it is written |
| Seeing and answering | Object detection on the iGPU while document Q&A answers on the NPU | 1 min | Two chips, no slowdown (measured above) |
| Review in seconds | Code review on the 30B model | ~30 s | 45 tokens a second |
| Reads what it sees | Screen OCR on a sample receipt | ~30 s | A 7B vision model on the iGPU |

About seven minutes a loop. With a camera, webcam effects joins the third
scene.

### 6. Steps

| Step | Delivers | Size |
| --- | --- | --- |
| **A** | Scenes: the data, start and stop, a row of scenario buttons on the home page (R21) | 3-4 days |
| **B** | The director, `/api/autodemo`, follow mode with captions and live figures, pause on input, keep-awake, the failure rules, the pre-start check | 3-4 days |
| **C** | Spoken captions, camera scenes, a recorded-clip source for the listening demos, start and stop by the clock | 2-3 days |

A long run on the real machine belongs to B: four hours watched through the
activity log before the first event.

### Tests

The director against stand-in runners: scenes play in order; a failing
scene is skipped and logged; three failures stop the loop; stopping
mid-scene leaves nothing running; a pause resumes at the next scene.

## To decide

1. **Captions only, or a voice too?**
2. **One demo on screen at a time**, with the hardware panel showing the
   others (proposed: nothing new to draw), **or a tiled stage view** of two
   or three at once (more to build, more to look at)?
3. **Is there a camera on the stand?** It decides whether the vision scenes
   show visitors or the screen.
4. **Which scenes, and in what order?** The table above is a proposal.
5. **Does a visitor get to take over**, or does the stand run locked?
