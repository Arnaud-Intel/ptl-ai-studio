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
  (Mixture-of-Experts, 30B total/~3B active per token, ~15.2GB), pinned to
  `GPU.1` (this dev machine's Arc B60). Already verified loadable on this
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
