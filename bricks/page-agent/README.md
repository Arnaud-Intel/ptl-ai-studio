# page-agent (experimental)

Turns a request -- one line, or a full brief -- into an illustrated,
self-contained web page that looks like somebody designed it, by putting
three models to work on three chips, with plain code conducting.

```
request ──> plan      a small model on the NPU names the page, says what it
                      presents and briefs six photographs
        ──> pictures  an image model takes them            ┐ at the same time
        ──> page      a coding model writes the HTML       ┘ with two GPUs
                      around their file names, told how the studio builds
                      a page
        ──> check     the conductor (code, on the CPU) puts its layout rules
                      into the page, verifies it and mends what it can
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
uv run page-agent "A landing page for a neighbourhood florist" --out florist.html
uv run page-agent --sample "Mountain bike" --keep-pictures ./pictures
uv run page-agent --list-samples
```

In the launcher: **Page Agent**. The panel shows the four steps with the chip
each is on, the plan as soon as it is written (the name, the headline, what is
on offer, and each picture with the place it is meant for), each picture as
soon as it is drawn, and the page as it is written.

### The samples

Six are briefs a studio could work from -- a fictional name, a look in a
palette and a kind of type, a headline, key figures, three things on offer
with their prices, a story, and one or two parts that give the page its
shape (opening hours as a table, trails with difficulty tags, a savings
estimator driven by a slider, a subscription in three price cards). One is a
single line, "A landing page for a neighbourhood florist.", to show that the
planner invents the rest and the page is built the same way.

Bakery, bike rental, solar installer, architecture studio, coffee roaster,
lake festival: they were chosen to look unlike one another (cream and
serif, glacier blue and sans, espresso-black, midnight blue), and for
subjects an image model draws well. A page for an app would need
screenshots, which it cannot draw.

## The plan

Thirteen labelled lines, which the planner fills in and plain code reads
(`plan.py`):

```
NAME, HEADLINE, STYLE, SECTIONS
HERO PHOTO
THING 1 / PHOTO 1, THING 2 / PHOTO 2, THING 3 / PHOTO 3
STORY PHOTO, DETAIL PHOTO
```

Each of the six photographs has a place on the page, and a size to suit it:
the first behind the headline (1024x576), three on the cards of what is on
offer (768x512), one beside the story (640x768), one behind the last call to
action (1024x576). The three things are named on the line just above their
photograph, which is what makes the card about a hardtail show a hardtail.

Code tidies the plan before anything is drawn: the parts of a description
that ask for writing are taken out (a banner "reading 'Les Heures Bleues'",
a clock that "reads 8:07" -- an image model cannot write), a picture that is
*of* writing (a ticket, a price list) is replaced by one made from the
thing's own name or dropped, the same sentence given for two pictures is one
picture, and the prompt's own example sent back as the answer is recognised
and asked for again.

## How the page is made to look designed

Left to itself the coding model answers every request with the same page: a
dimmed picture under a centred headline, then section after section of a
centred heading over three equal cards. `art_direction.py` is what changed
that, in two parts.

**What every request is given**, after the user's words and the plan: the
palette as seven CSS variables, two font stacks, and the page from top to
bottom as a numbered list -- navigation, hero, key figures, the three cards
with their pictures named, the story beside its picture, the request's other
parts each in the form that suits it (steps, a table, price cards, tags),
a quote, questions, a closing band, a footer. Advice ("give every section
its own layout") changed nothing; a specification was built as written.

**The layout rules every page gets.** Some twenty CSS rules -- the grid, the
split, the figures row, the card, the button, the dark band, the hero and
the closing band with their pictures behind a gradient -- are the brick's
own, put at the top of the page's stylesheet once it is written. They were
first given to the model to copy, because wherever a layout was only
described it came out wrong on every other page (figures stacked in a
column, a dark headline on a dark photograph); copied, three pages in
thirteen were still missing one. The model is shown the rules and told they are
there. Everything else is its own: the colours, the type, the navigation,
every component, the copy, the scripts -- and its rules come after these, so
it can overrule any of them.

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
code. It decides the order, loads and unloads, gives the page its layout
rules, and checks the result. It is not a fourth model.

## What the conductor checks, and what it does about it

| Check | If it fails |
| --- | --- |
| complete (ends with `</html>`) | the page is asked for once more |
| pictures placed (the first one, and all but one of the others) | asked for once more if the first is missing or fewer than half are there; otherwise shown as it is |
| each picture once | more pictures are drawn (below) |
| self-contained (nothing fetched from the network) | an `<img>` pointing elsewhere is given a picture or taken out (below); anything else is reported |
| fits a phone (viewport meta) | reported |
| has a main heading | reported |
| keeps the request's figures (two thirds of the prices, hours and numbers it gave) | reported, with the ones not found |

A second writing of the page is over a minute, so it is kept for a page that
cannot be shown. A page with five of its six pictures is a good page with a
picture to spare.

**A page that asks for a picture nobody drew gets one.** A page with three
trail cards as well as three bike cards has more places than pictures, and
the coding model fills them with what it has -- a planned picture again -- or
with a placeholder service's address (`https://placehold.co/300x200?text=...`),
which is a broken image offline. Every such `<img>` gets a file of its own,
drawn from the page's own alt text with the card's heading in front; one
with no alt text to draw from is left alone if it repeats a picture and
taken out if it points nowhere.

**A page that goes round in circles is stopped.** A long stylesheet can turn
into a loop -- the same few rules again and again until the tokens run out,
150 seconds for nothing. `runaway.py` watches how well the last 3,000
characters compress (a sound page: 19% of their size at the very least; a
loop: 4 to 6%), stops the page under 10%, and the page is asked for again
with that said. What it was writing is kept beside the pictures as
`stopped-page.html`.

## What it takes (XPS 14, 2026-10-08)

Core Ultra X7 358H, Arc B390 integrated GPU, NPU; an Arc Pro B60 attached for
the two-GPU rows. The seven samples, built one after the other in one
process.

| | First build (three models load) | Next builds |
| --- | --- | --- |
| integrated GPU + B60 (two runs of seven) | 145 s, 162 s | 82-92 s for ten of twelve; 103 s; 120 s for the one written twice |
| integrated GPU alone (three builds, **on a machine holding itself back**, see below) | 272 s | 287 s, 290 s |

Step by step, with two GPUs: a plan takes 13-15 s on the NPU (210 to 280
tokens at 19 tokens/s) after 7 s to load; FLUX draws the six pictures in 38
to 50 s on the integrated GPU after about 30 s to load (5 to 10 s each --
alone on the machine they take 4.5 to 8.5 s); the coding model writes a page
of 3,800 to 4,400 tokens in 68 to 79 s on the B60 (90 s once) after 42 s to
load, at 52 to 64 tokens/s over the page (45 once) -- it writes at 65 to 68
with the machine to itself, and less while the pictures are drawn beside
it.

On one GPU nothing stays loaded from one build to the next but the planner:
every build loads the image model (20 s), draws, then loads the coding
model (20 s) and writes.

These pages take about twice as long to build as the first version's (44 s
for a warm bakery page): they have six pictures instead of three and eight
sections instead of four, and the writing is nearly all of it.

**The single-GPU row, and anything timed after 21:00 that day, is slower
than the machine can do.** After two hours of builds back to back the
laptop began limiting itself: the processor package drew 15 W with eight
cores busy, the same as at idle, and everything that feeds a model slowed
with it -- the coding model on the B60 wrote at 32 to 38 tokens/s instead of
52 to 64, the integrated GPU at 25 to 31 where it had written at 40 that
morning, a picture took 11 to 20 s instead of 5 to 10. A two-GPU build took
130 to 140 s warm in that state, and the first build in the launcher 203 s.
A few idle minutes did not undo it. What a single-GPU build takes on a cool
machine was not measured again; the morning's shorter pages took 206 to
234 s.

## What was learned building it

- **The planner fills in nothing but sentences.** Asked for lines like
  `IMAGE name shape: ...` it sent back the words "name shape". File names and
  sizes are given out by the line's label, and the model writes only
  descriptions.
- **Ask a small model for a photograph, not for "a picture".** For a brief
  that went into colours, type and prices, "describe the picture" got "a
  trail map with EUR 39 / 210 in chalk white, in a tight modern sans".
  "A photograph, for a photographer who has not read the request and only
  reads English", with one example, got scenes -- and English ones for the
  one French request tried.
- **An example is copied now and then.** An architecture studio was planned
  six photographs of pottery, word for word the prompt's example. The
  example's lines are recognised in an answer, and the planner now takes its
  most likely word each time instead of drawing among the likely ones: nine
  sound plans out of nine that way.
- **The coding model can answer with an opening code fence and stop** --
  "```", then the end. Ending the request with "Start your reply with
  `<!DOCTYPE html>`" fixed it while requests were short; with the art
  direction in them, two in four came back empty again. The answer is now
  begun for the model: the chat template is applied by hand, the doctype put
  after it, and the runtime asked to continue (`begin`, in doc-qa's model
  wrapper, which html-creator uses for every page). None in the 48 pages
  written since.
- **A long stylesheet loops at a low temperature.** At the 0.2 every brick
  draws at, three pages in fifteen ran away into the same rules repeated;
  at the coding model's own 0.7, none of thirteen written from the same
  requests. The page is written at 0.7; the watch is for the rest, and
  stopped one page of the twenty-two built since -- 54 s in, where the
  stylesheet was long finished. What it was repeating was not kept that
  time; it is now.
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

- The planner is the weakest model in the chain. Its photographs are plain
  ("a single coffee bean on a wooden tray" for the top of a launch page) and
  now and then beside the point (a clerk at a desk for a bike workshop).
  Nobody checks a picture against its description: what the image model drew
  is what the page gets.
- The planner reads a brief's wording closely. "A launch page for Kivu
  Nightfall, the new coffee of the roaster Braise & Co." with three coffees
  listed got "Roasted after dark, How to Brew, Subscription" as its three
  things; "a one-page site for Braise & Co., a coffee roaster" with the same
  list got the three coffees. The sample was reworded. The same request
  always gets the same plan, so a brief can be tried before a show.
- Pictures still get lettering now and then, and it is gibberish. The plan
  is cleaned of what asks for it; an image model adds some unasked.
- The page does not always carry everything a long brief lists -- week
  prices beside day prices, the year of each project. The last check says
  which figures are missing; nothing puts them back.
- Every page has the studio's skeleton: the same order of parts, the same
  twenty layout rules. The palette, the type and the content make them look
  unlike one another, not the structure.
- Sections, copy and scripts vary from one build to the next, since the page
  is written at a temperature that lets it. The same request gives the same
  plan and the same pictures, and another page.
- A build is a minute and a half once the models are loaded, and nothing
  lets a presenter ask for a shorter page.
- Extra pictures are drawn after the page, so on one GPU the coding model is
  unloaded to draw them and loads again for the next build.
- Stopping a build (the hardware panel's close button) stops the page being
  written and any pictures not yet drawn; the page comes back incomplete and
  says so.
- **A process that has built a page ends itself abruptly.** After a
  single-GPU build with extra pictures, the process did not exit: every
  model was released, Python had torn down every module, and one thread then
  spun at 100% inside the runtimes' own teardown -- five times out of five,
  fourteen minutes when it was first left alone. Eight shorter sequences of
  the same loads and releases exited at once, so what sets it off is not
  known. The command-line tool, and the launcher once a page has been built
  in it, end the process outright when their own work is done
  ([`leaving.py`](src/page_agent/leaving.py)); on Windows that takes
  `TerminateProcess`, because `os._exit` hung the same way.
