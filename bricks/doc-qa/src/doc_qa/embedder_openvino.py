"""OpenVINO embedding backend: runs an embedding model on Intel CPU/iGPU/NPU
via `openvino_genai.TextEmbeddingPipeline`.

Requires this brick's `openvino` extra. Downloads Intel's pre-converted
Qwen3 embedding model from Hugging Face by default; pass `model_dir` to use
one you converted yourself.
"""
from __future__ import annotations

from typing import Callable

from pantherlake_ai_core import npu
from pantherlake_ai_core.engine import ov_config_for
from pantherlake_ai_core.model_cache import resolve_snapshot

_DEFAULT_REPO = "OpenVINO/Qwen3-Embedding-0.6B-int8-ov"

# How a text becomes one vector. Qwen3-Embedding is trained to be read at
# its last token; the pipeline's default reads the first. Measured on the
# sample folder with ten questions whose answer is in one known file
# (2026-10-09): the right file came first for 2 of 10 with the default, and
# for 8 of 10 read at the last token -- on the integrated GPU and on the NPU
# alike. With the default, the search was close to drawing passages by lot,
# and the language model was answering from excerpts beside the point.
LAST_TOKEN = "last-token"
FIRST_TOKEN = "first-token"  # the pipeline's default: what indexes built before 2026-10-09 hold

# How alike a passage has to be to a question to be worth reading for it
# (cosine, with the model read at its last token). Measured on the sample
# folder, 2026-10-09: the best passage scored 0.37 to 0.73 for ten questions
# the documents answer, 0.19 to 0.34 for six about the same company that
# they do not, 0.13 to 0.20 for four about something else. Set under the
# lowest right one with room to spare: a question wrongly turned away costs
# more than one the model is left to decline by itself.
RELEVANCE_FLOOR = 0.30

# The NPU runs a graph of one fixed shape: one text at a time, padded to this
# many tokens (a passage is 900 characters, some 250 tokens of prose). Left
# to pad to the model's whole window it took 1.6 s a text; at this length
# 0.07 s. The padding goes on the left, which is where a model read at its
# last token needs it -- but only together with a fixed length: on the left
# without one, every vector came back as not-a-number.
NPU_MAX_TOKENS = 512


class OpenVINOEmbedder:
    def __init__(
        self,
        device: str = "AUTO",
        model_dir: str | None = None,
        on_downloading: Callable[[], None] | None = None,
        pooling: str = LAST_TOKEN,
    ):
        import openvino_genai as ov_genai

        resolved_dir = resolve_snapshot(_DEFAULT_REPO, local_dir=model_dir, on_downloading=on_downloading)
        self.device = device
        # Two indexes can share their vectors only if this is the same: it
        # goes into the key an index is kept under.
        self.identity = f"qwen3-embedding-0.6b/{pooling}"
        # Scores made the old way were never measured, and mean something else.
        self.relevance_floor = RELEVANCE_FLOOR if pooling == LAST_TOKEN else None
        self._one_at_a_time = npu.is_npu(device)
        config: dict = {}
        if pooling == LAST_TOKEN:
            config.update(pooling_type=ov_genai.TextEmbeddingPipeline.PoolingType.LAST_TOKEN, normalize=True)
            if self._one_at_a_time:
                config.update(padding_side="left", batch_size=1, max_length=NPU_MAX_TOKENS, pad_to_max_length=True)
        # npu.guard: on the NPU, one brick's request at a time, and none at
        # all once Windows has reset the chip (see core's npu module).
        with npu.guard(device):
            self.pipeline = ov_genai.TextEmbeddingPipeline(resolved_dir, device, **config, **ov_config_for(device))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._one_at_a_time:
            # A call with two texts fails inside the NPU plugin ("padded_shape[i]
            # == position_shape[i]"): indexing any folder of more than one
            # passage did, on that chip. Each text takes its own turn, so
            # another brick's request can pass between two.
            vectors = []
            for text in texts:
                with npu.guard(self.device):
                    vectors.append(self.pipeline.embed_documents([text])[0])
            return vectors
        with npu.guard(self.device):
            return self.pipeline.embed_documents(list(texts))

    def embed_query(self, text: str) -> list[float]:
        with npu.guard(self.device):
            return self.pipeline.embed_query(text)
