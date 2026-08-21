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
> folder doesn't re-embed everything.

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

Force a full rebuild of the index (e.g. after editing the documents):

```bash
uv run doc-qa ./my-notes --reindex
```

## Options

| Flag | Description |
| --- | --- |
| `folder` | Folder of `.txt`/`.md`/`.pdf` files to index and answer questions about. |
| `--engine {portable,openvino}` | Inference backend. Default: `portable`. |
| `--compute-device NAME` | `openvino` engine only: `AUTO`, `CPU`, `GPU`, `NPU`. Ignored for `portable` (CPU only). |
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
3. **Retrieve & answer** ([`pipeline.py`](src/doc_qa/pipeline.py)) — the
   question is embedded the same way, the top-k most similar chunks are
   retrieved, and a local chat model answers from those excerpts only (the
   system prompt tells it to say "I don't know" rather than guess, and to
   cite which excerpt(s) it used).

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
maximum answer quality. To use a different chat/embedding model, edit the
`_DEFAULT_REPO`/`_DEFAULT_FILENAME` constants in `llm_portable.py` /
`embedder_portable.py`, or pass `model_dir=` to `OpenVINOLLM`/
`OpenVINOEmbedder` for a model you converted yourself with
`optimum-cli export openvino`.

## Notes / current limitations

- Answer quality reflects the small default models -- they can be terse or
  occasionally miss nuance. Point `PortableLLM`/`OpenVINOLLM` at a bigger
  model (e.g. `Qwen2.5-7B-Instruct`) if you have the RAM/VRAM for it.
- Retrieval is a flat top-k cosine search with no re-ranking. `openvino_genai`
  ships a `TextRerankPipeline` that would be a natural next step if
  precision on larger document sets becomes an issue.
- One session (one engine/device/index) at a time in the launcher UI.
