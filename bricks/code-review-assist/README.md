# Commit & Code Review Assistant

Reads a git diff and asks a local LLM to draft a commit message and review
notes. Composes `doc-qa`'s `engine_factory.create_llm` -- no LLM code of
its own, same pattern `meeting-notes` uses for its transcriber and LLM.

## Model choice

`doc-qa`'s own default LLM (`OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov`) is a
small general-purpose chat model -- fine for document Q&A, noticeably
weaker at reading a diff and reasoning about code. This brick asks
`create_llm` for a different, coding-specialized model via its new
`model_repo` override instead:

- **OpenVINO engine**: `OpenVINO/Qwen3-Coder-30B-A3B-Instruct-int4-ov`, a
  Mixture-of-Experts coding model -- 30B total parameters, ~3B active per
  token, ~15.2GB on disk. MoE is the point here: a general-purpose small
  model is weak at code review, but a large *dense* coding model would be
  slow token-by-token on a single GPU. MoE gives large-model code knowledge
  at small-model decode cost.
- **Portable engine**: `Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF` -- matches
  doc-qa's own 1.5B-general-model sizing for its CPU fallback, just
  coding-flavored. `n_ctx` is raised to 8192 (from doc-qa's 4096 default)
  since a real diff plus prompt routinely needs more room.

### Verified on this machine

`Qwen3-Coder-30B-A3B` is `qwen3_moe` architecture -- newer than the `qwen2`
architecture doc-qa's own default model uses, so this wasn't assumed to
just work. Verified directly before writing any brick code: downloaded the
model, loaded it via `openvino_genai.LLMPipeline(model_dir, "GPU.1",
CACHE_DIR="ov_cache")`, and generated a real completion -- loaded and ran
without error on this machine's OpenVINO 2026.3.0. Confirmed again through
the full brick (both the CLI and the launcher UI, both diff sources): diffs
against this repo's own working tree produced a properly formatted
conventional-commit message and specific, line-referenced review notes
(including correctly noticing when the diff had been truncated, rather
than inventing content for the missing part) in ~60 seconds total
(compile + two generations) on the Arc B60. No fallback was needed. If a
future `openvino_genai` version regresses `qwen3_moe` support, the
documented fallback is `OpenVINO/Qwen2.5-Coder-14B-Instruct-int4-ov`
(dense, `qwen2` architecture -- same family as doc-qa's proven-working
default, comfortably fits the B60's 24GB either way).

## Device

The OpenVINO engine defaults to `GPU.1` (this dev machine's Arc B60,
24GB VRAM -- the model needs ~15GB, comfortable headroom), not `AUTO` the
way every other brick's OpenVINO default is. `GPU.1` is this specific
machine's card id, not a portable convention -- override with
`--compute-device` (CLI) or the compute-device dropdown (launcher) on a
machine without that exact device. The model isn't split across both
GPUs: `openvino_genai.LLMPipeline` targets a single device string, and
there's no good way to shard one model's weights across the
memory-bandwidth-limited, system-RAM-sharing iGPU and the dGPU. Two GPUs
are more useful for running two different bricks concurrently, one per
device.

## Diff handling

- `--folder` runs `git diff --no-color --no-ext-diff <against>` in that
  working tree (default `--against HEAD`, i.e. all uncommitted changes);
  `--diff-file` reads a diff from a file instead. The subprocess call has
  a 20s timeout.
- A diff over 12,000 characters is truncated before it reaches the model;
  the result reports the original character count and whether truncation
  happened, so the CLI/UI can warn that some changes may not be reflected.
- An empty diff raises before any LLM call.

## Two LLM calls, not one

The commit message and review notes come from two separate `.answer()`
calls with distinct system prompts, not one call with parsed/split output
-- avoids inventing a fragile text-splitting layer for a small local
model's not-always-consistent formatting (the lesson `expense-extract`'s
`parsing.py` already learned the hard way, for JSON specifically). Costs
roughly double the latency on what's already an infrequent, non-live
action.
