# Auto Demo

Status: **it runs, on a stage of its own** (2026-10-09). The app opens on a
start screen with two ways in, *Auto Demo* and *Manual demo*; the Auto Demo
plays a playlist of three scenes in a loop, each told as a short story. The
rest of this note, from "What a passer-by sees" down, is the design it was
first built from, written 2026-10-04.

To start it: open the app, choose **Auto Demo**. The dialog says what the
stand has and what one turn of the loop will play, and takes four choices:
the language the story is told in (English or French), whether to use the
discrete GPU, larger text for a big display, and full screen.

## Where it stands (2026-10-10)

Decided with the user, 2026-10-08 to -10:

| Question | Answer |
| --- | --- |
| Captions or a voice? | Captions only. A stand at a large event is too loud for sound |
| Which language? | English unless French is chosen when the loop is started. Every sentence is written in both |
| What is on screen? | A stage made for it and nothing else: **the caption on top, always**, telling the demo in hand as a story, a sentence or two at a time (what, why, how); **the demo's outputs** in the middle; **the chips** down the right. No settings, no options, no scrolling |
| A camera on the stand? | Almost always. The person starting the loop can leave it out (a meeting, a film for colleagues): a scene that would show it plays a video kept on the machine instead, never the screen |
| The discrete GPU? | Not always there: looked for when the loop starts, and the person starting it can leave it out |
| A large display? | Sometimes: a setting of the loop, not a different build |
| Somebody touches the machine? | A popup asks: keep playing, pause, or stop. Stop goes back to the start screen |
| Which scenes? | Page Agent, Expense extraction on the worn receipts, the two counting scenes in turn, the Video Commentator, and Detection with Q&A now that both bricks have had a proofing pass. **Which of them play is ticked when the loop is started** (asked for on 2026-10-10), and remembered |

Three things were put to the user as changes to what was asked, and built
that way:

- **The story follows the work, not a clock.** A sentence is due when the
  step it talks about is seen at work (the planner, the pictures, the page),
  and one that points at something on screen ("that is the code appearing")
  when that step has a figure to show, which is after its model has loaded.
  A model that takes forty seconds to load one day takes fifteen the next;
  a timed script would talk about pictures nobody can see yet.
- **The popup answers itself.** Left alone for fifteen seconds it closes and
  the loop carries on (or stays paused, if it was). A pause nobody ends
  ends after five minutes. A stand that waits for an answer from somebody
  who has walked away has stopped for the day.
- **A pause interrupts nothing.** The scene in hand runs to its end and its
  result stays on screen for as long as the loop is paused; the next scene
  does not start. Stopping a page build half-way to honour a pause would
  throw away the thing the person stopped to look at.

Built, and tested without a browser or a model (`tests/test_autodemo.py`):

- **Scenes as data** (`launcher/autodemo.py`): the routes a scene calls and
  what it waits for; its story as `Beat`s (the text, and when it is due:
  seconds into the scene, a stage of the demo seen at work, that stage
  having a figure, the result being in); one `Chip` line per chip (what it
  does, with which model, and whose live figure goes on its line); which of
  the stage's views draws it, and what that view needs.
- **The director** (same file): a thread in the launcher that plays the
  playlist in a loop through the launcher's own routes, the way a person
  would. A scene that fails is noted and skipped; three in a row stop the
  loop with a notice. `pause()` holds the loop, `resume()` or five idle
  minutes release it, `skip()` ends the scene in hand. Whatever a scene
  started is stopped when it ends, however it ends.
- **The playlist** (`launcher/autodemo_scenes.py`): each scene is built for
  the stand it plays on -- with the discrete GPU or without, NPU or not --
  in the language chosen, and takes another sample each turn of the loop.
  Every scene has a key the start screen ticks it by. Scenes that share a
  place in the loop (the two counting scenes) take turns in it; if one was
  left out, or cannot play here, the other has the place every turn.
  `HELD_BACK` is empty, and stays for the next scene written before its
  demo is ready.
- **The routes**: `GET /api/autodemo` (state, scene, and every scene of the
  playlist with whether it was chosen, can play, and plays this turn),
  `GET /api/autodemo/check` (the same before starting), `GET
  /api/autodemo/result`, and `POST .../start` (which takes `scenes`, the
  keys to play, and `camera`: "auto" or "off"), `stop`, `pause`, `resume`,
  `skip`.
- **Keep-awake** while the loop runs (`launcher/keep_awake.py`), and no
  upgrade while it is on.
- **The page's side** (`static/app.js`, "Start screen, and the Auto Demo's
  stage"). Addresses: `#/` the start screen, `#/stage` the stage, `#/demos`
  the grid, `#/brick/<id>` a demo. A reload stays where it was; a page
  opened while the loop runs joins it. The stage has one output view per
  kind of demo, each made to fit whole: the page agent's plan, six pictures
  as they are drawn, the page's code as it is written, then the page itself,
  scrolled slowly from top to bottom; the receipts, each beside the line
  the two models make of it; two camera pictures with what was found drawn
  on them, their chip, frame rate and counts; the commentator's video with
  its line on it as a subtitle, what was actually seen under it, and the
  lines said so far with the voice of each; the detector's picture, with
  what is in it and how fast, beside a question and its answer as it is
  written. The chips column shows each chip's load, what it does in this
  demo, and its own figure.
- **Before it starts**: the dialog lists what the stand has, then every
  scene with a tick -- "plays", "left out", "plays every other turn, taking
  turns with ...", or why it cannot play here, in which case it cannot be
  ticked. What is remembered is what was taken out, so a scene added later
  plays without anybody having to find it.

**Watched runs** (XPS 14 with the B60, 2026-10-09, in a headless Edge that
photographs every sentence and measures what overflows):

- A whole turn in English: the page built and shown in 2 min 23 s with its
  three models loading (planner 18.6 tok/s on the NPU, 7 images a minute on
  the integrated GPU while the 30B coder wrote at 43 to 52 tok/s on the
  B60), the receipts, two London cameras at 25 frames a second each on the
  integrated GPU and the NPU. The popup left unanswered closed itself; a
  pause asked for while the receipts were read let them finish and held
  their result past the scene's own time; resume went on to the cameras;
  stop went back to the start screen with nothing left running. No scene
  failed.
- A second turn in French, with the page scene skipped part-way and the
  five worn receipts.
- Nothing overflowed and no sentence had to be set smaller at 1280x680,
  1440x810 and 1920x1080.

**The receipts scene showed what the receipts brick got wrong.** With each
receipt beside its line, the first run put "Meridian Robotics" -- the
customer, after "Billed to:" -- as the vendor of all three, lost two dates
and filed a cafe under Lodging. Measured on the fourteen sample receipts
that have a checked answer (Qwen2.5-1.5B on the NPU): vendor 5 right of 14,
date 8, category 7, amount 13. Fixed in `expense_extract/pipeline.py`: the
model is no longer shown the buyer's line, a date printed once in figures
is read by rule (day first, where the model read "12/09/2026" on a French
receipt as December), and the prompt says in plain sentences what a vendor
is and what goes in which category. After: vendor 13, date 14, category 12,
amount 13 -- and the one wrong amount is a hotel total too faded for the
vision model to read, which the brick flags ("Amount could not be matched
to the receipt text"); the stage shows that flag on its line and leaves it
out of the total.

**The two scenes added on 2026-10-10**, watched in English at 1920x1080
and in French at 1280x680 (XPS 14, on battery, the B60 not plugged in), the
other scenes unticked and the camera left out. Nothing overflowed, no
sentence was set smaller, no scene failed, and nothing was left running or
loaded after the stop.

- **A video, watched and commented on** (80 s). The first plain sentence
  14 s after the scene starts, the two models loading until then; the
  voices wait for it rather than for a clock. Then a sports commentator, a
  nature documentary, an upbeat voice, each change answered within two
  seconds without the picture being read again: about a second to see
  (22 tok/s on the integrated GPU) and a second to say (46 tok/s on the
  NPU). It plays another sample video each turn. It is silent, like the
  rest of the loop; its comments are in English in either language, and
  its last sentence says that the small model embroiders -- on this run it
  put the herd "under a starry night sky" at midday.
- **Seeing and answering at once** (55 s). The detector on a video (or the
  camera) for fourteen seconds alone, then the folder indexed and the
  question asked on the NPU. The first version started both together: the
  answer was written before the detector had shown a frame, and the story
  about one not slowing the other was over before it could be seen. The
  frame rate held at 24 while the answer was written (41 tok/s); it dipped
  to 14 for a second and a half earlier, when the NPU's models were being
  loaded, which is the CPU's work.

What the Object Detection pass found and changed is in that brick's README.

Not built yet, most useful first:

- **A run of several hours** on the real machine, watched through the
  activity log. The loop has been round a few times, not a day.
- **A whole turn with all six scenes**, and the camera: the two new scenes
  were watched on their own, with a video in the camera's place. The
  webcam was run once through the detector for its frame rate (30), and
  its picture was not looked at.
- **French comments** in the commentator scene: the two models answer in
  English.
- "Count whatever passes" -- a cattle drive and a bottle capping line through
  the same detector, each counted for what it shows -- takes turns with the
  streets: the streets on the first turn of the loop, the herd and the line
  on the second, never both in one turn. If one of the two cannot play, the
  other plays every turn. It says so itself that its tallies run high.
- The street scene plays two videos kept on the machine when they have been
  fetched, Toronto and Tyumen, named on screen by their city and nothing
  more (the live London cameras otherwise, which need the internet and keep
  their own names).
- A camera clip that starts over takes its chip off the hardware panel for
  a second or two; the stage holds its "at work" for six seconds to cover
  it, the panel does not.
- Deleting the loop's own expense reports (they pile up: one per turn), and
  unloading the large models between scenes on a machine short of memory.
- The stage has not been looked at on a stand without the discrete GPU: the
  scenes for it are built and tested, not watched.
- Step C of the plan below (start and stop by the clock, more camera
  scenes).

It was built on backlog ticket **R21** (one-click stage scenarios): a scene
here is what R21 calls a scenario, and a row of scenario buttons on the home
page is now a small step.

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
