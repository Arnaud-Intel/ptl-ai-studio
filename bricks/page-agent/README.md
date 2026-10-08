# page-agent (experimental)

Turns a one-line request into an illustrated, self-contained web page by
putting three models to work on three chips, with plain code conducting.

```
request ──> plan      a small model on the NPU writes the brief and
                      describes the pictures
        ──> pictures  an image model draws them            ┐ at the same time
        ──> page      a coding model writes the HTML       ┘ with two GPUs
                      around their file names
        ──> check     the conductor (code, on the CPU) verifies the page and
                      mends what it can
```

**Experimental** means: it works, results vary from one build to the next,
and its shape may still change. Everything below was measured on one machine
on one day.

The only model here that no other brick already wraps is the image model.
The planner is the model [meeting-notes](../meeting-notes/README.md) writes
with, the coding model is [html-creator](../html-creator/README.md)'s, and the
page is written and its pictures embedded by html-creator's own session.

## Setup

OpenVINO only -- there is no portable image model.

```bash
uv sync --extra openvino
```

Three models, downloaded on first use (`uv run panther-lake-prefetch`, or
"Models" in the launcher's footer, fetches them ahead of a show):

| Step | Model | Size |
| --- | --- | --- |
| plan | Qwen3-8B (channel-wise build on the NPU) | 4.5 GB |
| pictures | FLUX.1-schnell int4 (Apache-2.0) | 9.2 GB |
| page | Qwen3-Coder-30B-A3B int4 | 17 GB |

## Usage

```bash
uv run page-agent "A landing page for a neighbourhood bakery that bakes everything on site" --out bakery.html
uv run page-agent --sample "Mountain bike" --keep-pictures ./pictures
uv run page-agent --list-samples
```

In the launcher: **Page Agent**. The panel shows the four steps with the chip
each is on, the plan as soon as it is written, each picture as soon as it is
drawn, and the page as it is written.

## Who works where

Left to the conductor (`conductor.assign`):

- **the planner** goes to the NPU, else the integrated GPU, else the CPU;
- **the coding model**, the large one, goes to the discrete GPU if there is
  one, else the integrated GPU;
- **the image model** goes to the integrated GPU.

So with a discrete GPU each model has a chip, and the pictures are drawn
*while* the page is written: the page is written from the pictures'
descriptions and file names, never from the pictures, and waits for them only
when it is time to put them in. With one GPU the two take turns on it --
pictures first -- and each is unloaded before the other is loaded: a 30B
coding model and an image model do not belong in the same memory. Any of the
three can be put on another chip by name (`--planner-device`,
`--image-device`, `--page-device`; the three menus in the launcher).

"Supervised by the CPU" is literal and modest: the conductor is ordinary
code. It decides the order, loads and unloads, and checks the result. It is
not a fourth model.

## What the conductor checks, and what it does about it

| Check | If it fails |
| --- | --- |
| complete (ends with `</html>`) | the page is asked for once more, more compact |
| pictures placed (every planned one) | the page is asked for once more, with that said |
| each picture once | more pictures are drawn (below) |
| self-contained (nothing fetched from the network) | reported |
| fits a phone (viewport meta) | reported |
| has a main heading | reported |

**A picture shown several times is a page asking for more pictures.** The plan
has three; a page with three bike cards and three trail cards has more places
than that, and the coding model fills them with what it has. Telling it not to
did not work, and asking for the page again costs a minute and loses the cards
their pictures. So every repeated `<img>` gets a file of its own, drawn from
the page's own alt text with the card's heading in front ("Valley Cruiser X1:
close-up of a mountain bike...") -- four extra pictures in 18 s.

## What it takes (XPS 14, 2026-10-08)

Core Ultra X7 358H, Arc B390 integrated GPU, NPU; an Arc Pro B60 attached for
the two-GPU rows. One or two builds per row.

| | First build (three models load) | Next builds |
| --- | --- | --- |
| integrated GPU + B60, bakery page | 88 s | 44 s |
| integrated GPU + B60, bike page (a long one, four extra pictures) | 134-157 s | not measured |
| integrated GPU alone, bike page (the same) | 220-234 s | 206 s |

Step by step: a plan takes 9-11 s on the NPU (20 tokens/s) after 6 s to load;
FLUX draws 1024x576 in 6.5-8.5 s and 768x512 in 4.5-5.5 s on the integrated
GPU after 27 s to load (it holds about 13 GB of shared memory); the coding
model writes at 66-70 tokens/s on the B60 and 40 on the integrated GPU, after
40 s and 18 s to load, and a page is 2,300 to 4,400 tokens.

On one GPU nothing stays loaded from one build to the next but the planner:
every build loads the image model, then the coding model, and the image model
once more if the page needs extra pictures. That is most of the difference
with two GPUs; the rest is the coding model writing at 40 tokens/s instead of
66.

## What was learned building it

- **The planner fills in nothing but sentences.** Asked for lines like
  `IMAGE name shape: ...` it sent back the words "name shape". File names and
  sizes are now given out by position, and the model writes only
  descriptions.
- **The coding model can answer with an opening code fence and stop** --
  "```", then the end, on a freshly loaded model, sampled or not. Ending the
  request with "Start your reply with `<!DOCTYPE html>`" fixed it, four times
  out of four. The check-and-retry had caught it before the cause was known.
- **"Use each picture once" was not obeyed**, said before the list of
  pictures or after it, whenever the page had more cards than pictures. That
  is what led to drawing more pictures instead.
- **FLUX over the lighter candidate.** LCM Dreamshaper v7 draws a picture in
  under a second, but soft and loosely: asked for croissants on a counter it
  drew a room with bread-like shapes. FLUX drew the croissants.
- **Two pictures with one description need two seeds**, or they are the same
  picture; each takes its seed from its file name, so a request also draws
  the same pictures again.

## Known limits

- The planner is the weakest model in the chain. It was told to describe
  pictures in English whatever the language of the request, and described
  them in French for a French request.
- Pictures still get lettering now and then (a map, a sign), and it is
  gibberish: an image model cannot write, and the plan only asks it not to.
- The page is as long as the coding model makes it: 30 s to over a minute of
  writing. Nothing bounds it short of the 6,144-token limit, where it is
  asked for again, more compact.
- Extra pictures are drawn after the page, so on one GPU the coding model is
  unloaded to draw them and loads again for the next build.
- Stopping a build (the hardware panel's close button) stops the page being
  written and any pictures not yet drawn; the page comes back incomplete and
  says so.
- No picture is ever checked against its description: what the image model
  drew is what the page gets.
- **A process that has built a page ends itself abruptly.** After the
  single-GPU bike build above, the process did not exit: every model was
  released, Python had torn down every module, and one thread then spun at
  100% inside the runtimes' own teardown -- five times out of five, fourteen
  minutes when it was first left alone. Eight shorter sequences of the same
  loads and releases exited at once, so what sets it off is not known. The
  command-line tool, and the launcher once a page has been built in it, end
  the process outright when their own work is done
  ([`leaving.py`](src/page_agent/leaving.py)); on Windows that takes
  `TerminateProcess`, because `os._exit` hung the same way.
