"""Chatterbox-Turbo: the newer, much better-cloning voice model.

Four ONNX graphs run through ONNX Runtime, following Resemble AI's own
model card for `ResembleAI/chatterbox-turbo-ONNX` (MIT):

- **`speech_encoder`** -- the reference clip in, speaker conditioning out.
  It takes only audio, so it runs once at enrollment and every later
  synthesis reuses the result: enroll once, speak many times.
- **`embed_tokens`** + **`language_model`** -- a 350M Llama-family model
  that autoregressively predicts *speech* tokens for the text, conditioned
  on that speaker. This is the slow part: one forward pass per token.
- **`conditional_decoder`** -- speech tokens to a waveform in a single
  step (Resemble distilled it down from ten), so it costs a fraction of
  the token loop.

Why this model, and why in this shape -- all measured on this machine
against the OpenVoice path this brick already had:

- **It actually clones.** Against a real human reference clip (LibriSpeech,
  so neither model had home advantage), speaker similarity to the original
  speaker measured with a neutral ECAPA-TDNN verifier: **0.648 here vs
  0.287 for OpenVoice**. OpenVoice v1 only re-colors the timbre of its own
  fixed base speaker, so it cannot reproduce a real person's identity; this
  model is conditioned on the reference directly and does.
- **q4f16, not fp16.** Same graphs, same CPU: q4f16 is 536MB and runs at
  2.5x realtime; fp16 is 1584MB and runs at 14.1x realtime, because this
  CPU has no native fp16 compute and spends the difference casting.
- **ONNX Runtime, not OpenVINO.** OpenVINO reads the fp16 graphs happily
  but compiles them only for CPU -- the GPU and NPU plugins both reject
  their dynamic shapes -- and fp16-on-CPU is the slow path above. So this
  model is CPU-only, and OpenVoice stays the one that reaches the NPU and
  iGPU. That is the trade this brick now offers as a choice.
"""
from __future__ import annotations

import re
from typing import Callable

import numpy as np
from pantherlake_ai_core.model_cache import is_file_cached, resolve_file

REPO_ID = "ResembleAI/chatterbox-turbo-ONNX"
SAMPLE_RATE = 24000

# Quantization variant. See the module docstring: this is the fast one on
# a CPU, by a factor of five and a half.
DTYPE = "q4f16"
GRAPHS = ("speech_encoder", "embed_tokens", "language_model", "conditional_decoder")

# Vocabulary landmarks and cache geometry, from the model card.
START_SPEECH_TOKEN = 6561
STOP_SPEECH_TOKEN = 6562
SILENCE_TOKEN = 4299
NUM_KV_HEADS = 16
HEAD_DIM = 64

# The model's own generation ceiling. Roughly 30 seconds of speech, which
# is why `synthesize` splits long text into sentences rather than trusting
# one pass to reach the end of a paragraph.
MAX_NEW_TOKENS = 1024
REPETITION_PENALTY = 1.2

# Sounds the model can make on cue, written inline in the text. Native to
# Turbo -- not a filter bolted on afterwards.
PARALINGUISTIC_TAGS = ("[laugh]", "[chuckle]", "[cough]", "[sigh]", "[gasp]", "[sniff]", "[clear_throat]")


def _graph_files(dtype: str = DTYPE) -> dict[str, str]:
    """Repo-relative paths of the graph + external-weights pair per stage.

    Every graph keeps its weights in a sibling `.onnx_data`; ONNX Runtime
    opens that by name, so both have to be in the cache before the graph
    is loaded, even though only the `.onnx` path is ever passed anywhere.
    """
    suffix = "" if dtype == "fp32" else ("_quantized" if dtype == "q8" else f"_{dtype}")
    return {name: f"onnx/{name}{suffix}.onnx" for name in GRAPHS}


def resolve_files(
    local_dir: str | None = None,
    on_downloading: Callable[[], None] | None = None,
    dtype: str = DTYPE,
) -> dict[str, str]:
    """Local paths to the four graphs plus the tokenizer, downloading them
    on first use. `on_downloading` fires once, before the first fetch, and
    not at all when everything is already cached."""
    graphs = _graph_files(dtype)
    wanted = [*graphs.values(), *(f"{p}_data" for p in graphs.values()), "tokenizer.json"]

    if local_dir:
        from pathlib import Path

        root = Path(local_dir)
        missing = [rel for rel in wanted if not (root / rel).exists()]
        if missing:
            raise FileNotFoundError(f"{local_dir} is missing {len(missing)} Chatterbox file(s), e.g. {missing[0]}")
        return {**{name: str(root / rel) for name, rel in graphs.items()}, "tokenizer": str(root / "tokenizer.json")}

    if on_downloading is not None and not all(is_file_cached(REPO_ID, rel) for rel in wanted):
        on_downloading()

    resolved = {name: resolve_file(REPO_ID, rel) for name, rel in graphs.items()}
    for rel in graphs.values():  # weights: fetched for their side effect, opened by name
        resolve_file(REPO_ID, f"{rel}_data")
    resolved["tokenizer"] = resolve_file(REPO_ID, "tokenizer.json")
    return resolved


class ChatterboxModel:
    """One loaded copy of the four graphs. Enroll once, synthesize many."""

    def __init__(
        self,
        model_path: str | None = None,
        on_downloading: Callable[[], None] | None = None,
        dtype: str = DTYPE,
    ):
        import onnxruntime
        from tokenizers import Tokenizer

        paths = resolve_files(model_path, on_downloading, dtype)
        self.sessions = {name: onnxruntime.InferenceSession(paths[name]) for name in GRAPHS}
        self.tokenizer = Tokenizer.from_file(paths["tokenizer"])
        # float16 caches for the fp16-weighted graphs, float32 otherwise --
        # asked of the graph rather than assumed from `dtype`, since the
        # variants differ in which tensors they actually keep in half.
        self._cache_spec = {
            i.name: np.float16 if i.type == "tensor(float16)" else np.float32
            for i in self.sessions["language_model"].get_inputs()
            if "past_key_values" in i.name
        }

    # --- enrollment ---------------------------------------------------------

    def enroll(self, reference_audio_path: str) -> dict:
        """The speaker conditioning for one reference clip.

        The speech encoder takes audio and nothing else, so this is the
        whole of what a voice "is" to this model -- computed once here
        instead of on every sentence.
        """
        import librosa

        audio = librosa.load(reference_audio_path, sr=SAMPLE_RATE)[0]
        if audio.size == 0:
            raise ValueError("That reference clip is empty -- record a few seconds of speech first.")
        audio = audio[np.newaxis, :].astype(np.float32)
        cond_emb, prompt_token, speaker_embeddings, speaker_features = self.sessions["speech_encoder"].run(
            None, {"audio_values": audio}
        )
        return {
            "cond_emb": cond_emb,
            "prompt_token": prompt_token,
            "speaker_embeddings": speaker_embeddings,
            "speaker_features": speaker_features,
        }

    # --- synthesis ----------------------------------------------------------

    def _speech_tokens(self, text: str, voice: dict) -> np.ndarray:
        """The autoregressive loop: text (plus the speaker) in, speech
        tokens out. One `language_model` pass per token, which is what
        makes this the expensive stage."""
        input_ids = np.asarray([self.tokenizer.encode(text).ids], dtype=np.int64)
        generated = np.array([[START_SPEECH_TOKEN]], dtype=np.int64)
        attention_mask = position_ids = past = None

        for step in range(MAX_NEW_TOKENS):
            embeds = self.sessions["embed_tokens"].run(None, {"input_ids": input_ids})[0]
            if step == 0:
                embeds = np.concatenate((voice["cond_emb"], embeds), axis=1)
                batch, length, _ = embeds.shape
                past = {n: np.zeros([batch, NUM_KV_HEADS, 0, HEAD_DIM], dtype=t) for n, t in self._cache_spec.items()}
                attention_mask = np.ones((batch, length), dtype=np.int64)
                position_ids = np.arange(length, dtype=np.int64).reshape(1, -1).repeat(batch, axis=0)

            logits, *present = self.sessions["language_model"].run(
                None,
                dict(inputs_embeds=embeds, attention_mask=attention_mask, position_ids=position_ids, **past),
            )
            input_ids = np.argmax(_penalize(logits[:, -1, :], generated), axis=-1, keepdims=True).astype(np.int64)
            generated = np.concatenate((generated, input_ids), axis=-1)
            if (input_ids.flatten() == STOP_SPEECH_TOKEN).all():
                break

            attention_mask = np.concatenate([attention_mask, np.ones((batch, 1), dtype=np.int64)], axis=1)
            position_ids = position_ids[:, -1:] + 1
            for i, name in enumerate(past):
                past[name] = present[i]

        # Drop the start token, and the stop token when the loop found one
        # (a run that hit MAX_NEW_TOKENS has a real token in that slot).
        end = -1 if (input_ids.flatten() == STOP_SPEECH_TOKEN).all() else None
        return generated[:, 1:end]

    def synthesize(self, text: str, voice: dict) -> tuple[np.ndarray, int]:
        """Speak `text` in the enrolled voice.

        Long text is spoken a sentence at a time: one generation pass is
        capped at MAX_NEW_TOKENS (~30 seconds), so a paragraph handed over
        whole would simply stop partway through.
        """
        text = text.strip()
        if not text:
            raise ValueError("Nothing to say -- enter some text first.")

        pieces = []
        for chunk in split_sentences(text):
            tokens = self._speech_tokens(chunk, voice)
            silence = np.full((tokens.shape[0], 3), SILENCE_TOKEN, dtype=np.int64)
            speech_tokens = np.concatenate([voice["prompt_token"], tokens, silence], axis=1)
            wav = self.sessions["conditional_decoder"].run(
                None,
                dict(
                    speech_tokens=speech_tokens,
                    speaker_embeddings=voice["speaker_embeddings"],
                    speaker_features=voice["speaker_features"],
                ),
            )[0].squeeze(axis=0)
            pieces.append(np.asarray(wav, dtype=np.float32))

        return (pieces[0] if len(pieces) == 1 else np.concatenate(pieces)), SAMPLE_RATE


def _penalize(logits: np.ndarray, generated: np.ndarray, penalty: float = REPETITION_PENALTY) -> np.ndarray:
    """Discourage tokens already produced -- without it the model can lock
    into repeating one sound for the whole generation budget."""
    scores = np.take_along_axis(logits, generated, axis=1)
    scores = np.where(scores < 0, scores * penalty, scores / penalty)
    out = logits.copy()
    np.put_along_axis(out, generated, scores, axis=1)
    return out


# A sentence end, but not an abbreviation, a decimal, or one of the
# paralinguistic tags (which contain no periods, but do sit next to them).
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])")

# Long enough to be worth a pass of its own; below this a fragment is
# glued onto its neighbour rather than spoken as its own utterance, since
# very short generations lose the speaker's rhythm.
_MIN_CHUNK_CHARS = 24


def split_sentences(text: str, limit: int = 300) -> list[str]:
    """Text in speakable chunks: sentences, with short ones merged and
    over-long ones split on a comma so no single pass runs past the
    model's token budget."""
    chunks: list[str] = []
    for sentence in _SENTENCE_END.split(text.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        while len(sentence) > limit:
            cut = sentence.rfind(",", 0, limit)
            cut = cut + 1 if cut > limit // 3 else limit
            chunks.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        # Merge whenever either side is a fragment -- a stray "Yes." spoken
        # as its own pass comes out with none of the speaker's rhythm,
        # whether it lands before or after a full sentence.
        if chunks and (len(chunks[-1]) < _MIN_CHUNK_CHARS or len(sentence) < _MIN_CHUNK_CHARS):
            chunks[-1] = f"{chunks[-1]} {sentence}"
        else:
            chunks.append(sentence)
    return chunks or [text.strip()]
