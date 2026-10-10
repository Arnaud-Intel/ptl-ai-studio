# doc-qa

Ask questions about your own local documents — retrieval-augmented, fully
on-device. Point it at a folder of `.txt` / `.md` / `.pdf` files; it chunks
and embeds them into a local index, retrieves the most relevant chunks for
a question, and has a small local LLM answer using only those excerpts.

It supports two interchangeable inference engines:

- **`portable`** (default) — [llama.cpp](https://github.com/ggml-org/llama.cpp)
  via `llama-cpp-python`, using GGUF models. Runs anywhere, CPU only.
- **`openvino`** — [OpenVINO](https://docs.openvino.ai/) via `openvino_genai`.
  Targets Intel hardware explicitly: `CPU`, `GPU` (iGPU), or `NPU`.

Either way, nothing about your documents or questions leaves the machine.

## Setup

From the workspace root (`local_demo/`):

```bash
uv sync                    # portable engine only
uv sync --extra openvino   # also installs the OpenVINO engine
```

`llama-cpp-python` has no PyPI wheels, so this brick's `pyproject.toml`
points it at the project's own prebuilt CPU wheel index instead of falling
back to a from-source build (which needs `cmake` and a C++ toolchain).

> **First run note:** the first time you index a folder with a given
> `--engine`, its embedding and chat models are downloaded from Hugging
> Face and cached (`~/.cache/huggingface`). Every run after that is fully
> offline. The index itself is cached per folder+engine under
> `~/.cache/pantherlake-ai-studio/doc-qa/`, so re-running against the same
> folder doesn't re-embed everything -- as long as the folder holds what it
> held: add, remove or rewrite a file and the index is built again.

## Usage

```bash
uv run doc-qa ./my-notes --question "What did we decide about the launch date?"
```

Or ask multiple questions interactively (empty line / Ctrl+C to quit):

```bash
uv run doc-qa ./my-notes
```

Run on Intel NPU via OpenVINO:

```bash
uv run doc-qa ./my-notes --engine openvino --compute-device NPU
```

**What the documents change.** In the launcher, tick "Without the
documents" beside Ask and the question goes to the language model alone,
with nothing to read; untick it and ask again. On the sample folder, "who
makes the final go/no-go decision on the Lyon pilot, and on which date?"
gets "the team's captain, on September 16, 2023" from the model alone --
invented, and the same every time -- and "Priya Desai, on September 17"
with the documents, which is what the decision log says. Other questions
it declines alone ("I don't have that information"). The Auto Demo's
"documents" scene is this contrast, played by itself.
(`DocQASession.ask_alone`; `POST /api/doc-qa/ask` with `alone: true`.)

Force a full rebuild of the index (a changed folder is noticed without it):

```bash
uv run doc-qa ./my-notes --reindex
```

## Options

| Flag | Description |
| --- | --- |
| `folder` | Folder of `.txt`/`.md`/`.pdf` files to index and answer questions about. |
| `--engine {portable,openvino}` | Inference backend. Default: `portable`. |
| `--compute-device NAME` | `openvino` engine only: `AUTO`, `CPU`, `GPU`, `NPU`. Ignored for `portable` (CPU only). |
| `--model {1.5b,8b}` | `openvino` engine only: which language model writes the answers. Default: `8b` if this machine has it, else `1.5b` (see [Two language models](#two-language-models)). |
| `--reindex` | Rebuild the index even if a cached one exists for this folder+engine. |
| `--top-k N` | Number of source chunks to retrieve per question. Default: `4`. |
| `--question TEXT` | Ask a single question and exit, instead of an interactive loop. |

## How it works

1. **Load & chunk** ([`documents.py`](src/doc_qa/documents.py)) — reads
   every supported file under the folder and splits it into overlapping
   ~900-character windows. Deliberately dependency-free (no tokenizer at
   chunking time), so chunking doesn't depend on which engine is selected.
2. **Embed & index** ([`store.py`](src/doc_qa/store.py)) — each chunk is
   embedded and stored as an L2-normalized row in a plain numpy matrix;
   retrieval is a single matrix-vector cosine-similarity multiply. No
   vector database -- this is small-scale (a folder of notes, not a
   corpus), so a numpy array is simpler and has zero extra dependencies.
   `VectorStore.build()` replaces the whole index in one shot, matching
   this brick's own folder-ingest shape; `.add()` instead appends to
   whatever's already there, for a consumer doing incremental/streaming
   ingestion -- [`smart-recall`](../smart-recall/README.md) is why it
   exists, indexing screen captures continuously rather than all at once.
3. **Retrieve & answer** ([`pipeline.py`](src/doc_qa/pipeline.py)) — the
   question is embedded the same way, the top-k most similar chunks are
   retrieved, and a local chat model answers from those excerpts only, each
   handed to it under the name of its file (the system prompt asks for full
   sentences, the file each fact comes from, and "the documents do not say"
   rather than a guess). With the OpenVINO engine, a passage that is not
   close enough to the question is not read, and a question no passage is
   close to is answered "the documents do not say" without asking the model:
   a small model handed passages beside the point improvises.

The OpenVINO embedding model (Qwen3-Embedding) is read at its last token,
which is how it was trained; read at its first, as it was until 2026-10-09,
the right file came first for 2 questions of 10 on the sample folder, against
8 of 10. On the NPU it takes one text at a time in a fixed shape of 512
tokens (0.07 s each); see [`embedder_openvino.py`](src/doc_qa/embedder_openvino.py)
for what was measured.

Both the embedder and the LLM are picked by [`engine_factory.py`](src/doc_qa/engine_factory.py)
behind a small `Embedder`/`LLM` protocol -- `embedder_portable.py` /
`llm_portable.py` vs. `embedder_openvino.py` / `llm_openvino.py` -- the
same one-module-per-backend pattern `live-translation` uses.

## Default models

| | Portable (llama.cpp) | OpenVINO |
| --- | --- | --- |
| Embedding | [nomic-embed-text-v1.5-GGUF](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF) | [Qwen3-Embedding-0.6B-int8-ov](https://huggingface.co/OpenVINO/Qwen3-Embedding-0.6B-int8-ov) |
| Chat | [Qwen2.5-1.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF) | [Qwen2.5-1.5B-Instruct-int4-ov](https://huggingface.co/OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov) |

These are small (~1B-scale) models chosen for a fast first run, not
maximum answer quality; on the OpenVINO engine a larger one writes the
answers when the machine has it (below). To use a different chat/embedding model, edit the
`_DEFAULT_REPO`/`_DEFAULT_FILENAME` constants in `llm_portable.py` /
`embedder_portable.py`, or pass `model_dir=` to `OpenVINOLLM`/
`OpenVINOEmbedder` for a model you converted yourself with
`optimum-cli export openvino`.

## Two language models

On the OpenVINO engine the answers can be written by Qwen2.5-1.5B or by
Qwen3-8B (`--model`, the **Model** menu in the launcher; `language_models.py`
says which build goes on which chip, for every brick that uses them). Left to
itself the brick takes the 8B **when it is already on the disk** -- Meeting
Notes and the Page Agent fetch it -- and the 1.5B otherwise: it does not start
a download of 4.5 GB because somebody pressed Ask. Named, a model that is
missing is fetched first. The model can be changed between two questions; the
index is kept.

Why the larger one, measured on the sample folder with both on the XPS 14's
NPU (2026-10-10, on battery, one run each):

| | Qwen2.5-1.5B | Qwen3-8B |
| --- | --- | --- |
| Ten short questions with an answer in one file | every fact asked for; the deposit and the balance given in dollars where the file says euros | every fact but one figure (the 95% without the 57 of 60); the file named each time |
| Five with no answer in the files (two never reach a model: no passage is close) | of the other three, two declined and one answered with another product's date | all three declined, that one with the reason ("the October 19 date relates to the cold-storage variant") |
| The four sample questions, several parts each | two clean. Of the Dock C stop: the safety layer "failed" (it is what stopped the robot), a unit-test pass closes the case (the file says it does not) | two clean; that one right, in four bullets with their sources. One sentence breaks off into the next; refunds "should not be subtracted" (the policy subtracts them) |
| A short answer | 0.8 s, first word after 0.3 s, 55 tokens/s | 4.0 s, first word after 1.3 s, 16 tokens/s |
| A sample question's answer | 2 to 3 s | 8 to 13 s |

So the 8B is not right every time either; what it gets wrong is smaller, and
it does what the instructions ask (it names its files). Its build for the NPU
was quantised without calibration data, which is the likely reason for the
sentence that breaks off (see `meeting_notes.session`, and BACKLOG R36).

Asked *without the documents* (the tick in the launcher), the 8B makes things
up as freely as the 1.5B: of the Lyon pilot's decision it answered with the
President of the United States. That is what the tick is there to show; the
Auto Demo's scene keeps to the 1.5B, by name.

## Notes / current limitations

- Answer quality reflects the small default models -- they can be terse or
  occasionally miss nuance, and the 1.5B states things the files do not say
  (the table above). The portable engine has the 1.5B only.
- Retrieval is a flat top-k cosine search with no re-ranking. `openvino_genai`
  ships a `TextRerankPipeline` that would be a natural next step if
  precision on larger document sets becomes an issue.
- One session (one engine/device/index) at a time in the launcher UI.
