# HTML Creator

Generates a single self-contained HTML page -- a landing page from a text
prompt, or a styled summary report from a folder of documents (replacing a
rigid PDF for sharing a summary) -- using a local LLM. Composes `doc-qa`'s
`engine_factory.create_llm`, same pattern as `code-review-assist` -- no LLM
code of its own.

## Model choice

Reuses `code-review-assist`'s exact model choices and reasoning: generating
HTML/CSS is a code-generation task, so it asks `create_llm` for the same
coding-specialized model instead of doc-qa's small general-purpose default.

- **OpenVINO engine**: `OpenVINO/Qwen3-Coder-30B-A3B-Instruct-int4-ov`
  (Mixture-of-Experts, 30B total/~3B active per token, ~15.2GB), defaulting
  to the machine's discrete GPU when it has one (`GPU.1`, the Arc B60, on
  this dev machine) and to the integrated GPU otherwise (~38 tokens/s on
  the XPS 14's Arc B390; see `code-review-assist/README.md`). Already verified loadable on this
  machine this session (see `code-review-assist/README.md`) -- not
  re-verified in isolation, but this brick's own end-to-end run confirmed
  it works for HTML generation specifically, not just prose.
- **Portable engine**: `Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF`, `n_ctx`
  raised to `16384` (code-review-assist used `8192`; a full HTML page's
  output alone can run several thousand tokens, on top of a capped
  document input).

### Verified on this machine

Ran both modes on both engines against real inputs:

- **Landing page, portable**: `--prompt "a landing page for a small coffee
  shop called Bramble & Bean"` produced a complete, valid, self-contained
  HTML page (2.3KB) in well under a minute on CPU.
- **Landing page, openvino/GPU.1**: same prompt with more detail produced a
  15KB page -- full nav, hero, about, menu grid, testimonials, contact,
  footer, a responsive media query, and working smooth-scroll JS -- in
  ~2 minutes including one-time model compile. Notably, asked not to use
  external images, it used inline `data:image/svg+xml` placeholders instead
  of just omitting them -- the "no external assets" instruction held even
  for content the model had to invent.
- **Document summary, portable**: ran against `code-review-assist/`'s own
  README and produced a valid HTML summary.
- Both engines needed the code-fence stripper on the very first real run --
  confirms this wasn't a speculative guard, it's a real, common behavior for
  "output raw code" instructions.
- A second, richer landing-page prompt (through the launcher UI, on
  GPU.1) hit `max_tokens` mid-generation: the model was still writing a
  testimonials section when it got cut off, with no closing tags and no
  closing code fence. `html_truncated` correctly caught and reported this.
  It also exposed a real gap in the fence-stripper: a truncated response
  still starts with an opening ` ```html ` fence but never reaches a
  closing one, so the original strip-a-matched-pair regex left the fence
  marker sitting in front of the DOCTYPE, which broke the preview. Fixed by
  also stripping a lone unclosed leading fence. `landing_page`'s
  `max_tokens` was raised from 4096 to 6144 in response to the same test.
  Not exhaustively stress-tested beyond these runs; if truncation still
  shows up in practice, raise `_MAX_TOKENS_BY_MODE` in `session.py` further.

## Pictures

A language model writes text: it cannot see a picture or write one. It can
place pictures it is told about. `--pictures FOLDER` (the launcher's
"Pictures the page may use" field) offers a landing page the images in a
folder -- png, jpg, webp, gif or svg, up to 12, subfolders ignored:

- `pictures.py` appends a `PICTURES` list to the request: each file name,
  its size, and its caption when the folder has a `captions.txt`
  (`file name: what it shows`, one per line, `#` for comments). Without
  captions the model only has the file names to go on, so name them well.
- The model references a picture by file name, in an `<img>`, a CSS
  `url(...)` or a script. Afterwards every reference is replaced by the
  picture itself as a `data:` URI, whatever path the model put in front
  of the name -- the page is still one self-contained file. Text that
  merely mentions a file (an `alt`, a caption) is left alone.
- A file over 1.5 MB is scaled down to 1600 px (needs Pillow, which the
  launcher has); without Pillow, or for an oversized SVG or GIF, it is
  left out and the result says so.
- `HtmlResult.html` has the pictures inside; `html_source` is the page as
  the model wrote it, which is what the launcher shows under "View raw
  HTML". `pictures_used` says which ones it placed.

`sample-data/pictures/nordlys-trails/` is a kit of seven fictional
landscapes drawn by `scripts/build_demo_pictures.py` (SVG, standard
library only, no downloaded photos).

## The preview is a sandbox

The launcher shows the page in an `<iframe sandbox="allow-scripts">`:
scripts run, but there is no storage, no cookies and no `alert()`. A page
that reaches for `localStorage` throws and stops working, so the system
prompt forbids those calls and asks for state to be kept in variables.
"Full screen" expands that same frame to the window (and the screen, where
the browser grants it); it never takes the page out of the sandbox.

## Writing a scenario that holds up

The bundled scenarios (`sample-data/catalog.json`) were tuned by running
them on the XPS 14's integrated GPU and checking what came back: figures
and text read from the page, the game stepped frame by frame through
start, pause, a cleared wall, game over and restart, the lightbox driven
with clicks and keys.

**The same prompt does not give the same page twice.** The model samples
(temperature 0.2 on top of the model's own sampling settings), and three
runs of one prompt gave three different pages. A tighter prompt narrows
what can vary; it does not remove it. Each bundled scenario passed on the
runs that were checked (one to three each), and an earlier, looser wording
of every one of them produced at least one page with a visible fault. So:
rehearse a scenario before showing it, and if a page comes out wrong,
Generate again. Decoding without sampling would make a rehearsed page the
page you get; it is not done yet (see the backlog Inbox).

What made the prompts hold up:

- **Say how, not only what, for the parts that break.** "A slow zoom on
  the hero picture" produced a page whose text zoomed too; "the picture
  sits on its own absolutely positioned layer and only that layer zooms"
  did not.
- **Give the state machine.** The game's first versions each got a
  different detail of start, pause and restart wrong. Naming the four
  states and giving the one input handler as code fixed that.
- **Pin sizes that code must agree on** (a fixed 800 by 600 canvas scaled
  with CSS, bricks 71 wide and 8 apart) rather than leaving two parts of
  the page to guess.
- **Number the requirements the model must not drop.** "Show a fictional
  notice" in the opening sentence was skipped on one run; as item 8 of the
  list it was not.
- **Prefer a board to a long scrolling page** when the content is numbers.
  The hardware infographic came out sparse and stacked as a scrolling page
  and tidy as a full-window grid, the format that also suits a wall
  display.
- **Stop when it is good.** Past a point, each extra instruction fixed one
  detail and unsettled another: the hardware board went through six
  wordings, and the one kept is the third.
- **End with "Keep the code compact."** The cap is 6,144 tokens; these
  scenarios use 3,300 to 4,750, which is a minute and a half to two and a
  half minutes on the integrated GPU (30 to 38 tokens/s, slower after
  several runs in a row).

## Document handling

`folder_input.py` reuses `doc_qa.documents.load_documents` (the same
.txt/.md/.markdown/.pdf loader doc-qa's own ingestion uses) and
concatenates every file into one pass, capped at `MAX_DOCUMENT_CHARS =
20_000` -- simpler than summarizing per-document and composing the
results, at the cost of being less robust on a very large folder. Revisit
with a per-document-then-compose approach if that turns out to matter in
practice; not built speculatively now.

## Two failure modes, two independent guards

- **Input truncation** (`source_truncated`): the diff/document was too long
  and got cut before reaching the model -- same shape as
  `code-review-assist`'s diff truncation.
- **Output truncation** (`html_truncated`): the model's own output didn't
  end with `</html>`, most likely because it hit `max_tokens` mid-document.
  This is a worse failure mode than input truncation -- a cut-off HTML
  document may not render at all -- so it's checked and reported
  separately, deterministically (a string check, not a guess).

Also guarded: `html_cleanup.strip_code_fence` deterministically strips a
markdown code fence if the model wraps its raw-HTML output in one despite
being told not to (see "Verified on this machine" above -- this fires in
practice, not just in theory).

## One LLM call, not two

Unlike `code-review-assist` (one call for the commit message, one for
review notes), this brick makes exactly one `.answer()` call per
`generate()` -- there's only one output artifact (the HTML document), not
two independent ones to keep from bleeding into each other.
