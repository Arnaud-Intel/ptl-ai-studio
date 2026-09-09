"""The Chatterbox-Turbo backend's pure parts: which engines a model is
allowed on, how long text is split into speakable chunks, and the sampling
guard. No model is loaded and nothing is downloaded.
"""
from __future__ import annotations

import numpy as np
import pytest
from pantherlake_ai_core.engine import Engine
from voice_clone_studio import chatterbox_model, engine_factory
from voice_clone_studio.pipeline import VoiceCloneSession


# --- which model runs where -------------------------------------------------


def test_chatterbox_refuses_the_engine_it_has_no_backend_for():
    """OpenVINO compiles these graphs for CPU only and fp16-on-CPU is the
    slow path, so this model is portable-only -- said up front, with the
    way out, rather than failing somewhere inside a compile."""
    with pytest.raises(ValueError, match="doesn't run on the 'openvino' engine"):
        engine_factory.create_cloner(Engine.OPENVINO, model="chatterbox")


def test_openvoice_still_takes_either_engine():
    assert engine_factory.MODEL_ENGINES["openvoice"] == (Engine.PORTABLE, Engine.OPENVINO)
    assert engine_factory.MODEL_ENGINES["chatterbox"] == (Engine.PORTABLE,)


def test_an_unknown_model_is_named_in_the_error():
    with pytest.raises(ValueError, match="Unknown model 'bogus'"):
        engine_factory.create_cloner(Engine.PORTABLE, model="bogus")


def test_the_better_cloning_model_is_the_default():
    assert engine_factory.DEFAULT_MODEL == "chatterbox"


# --- a style asked of a model that has none ---------------------------------


class FakeCloner:
    supports_styles = False

    def __init__(self):
        self.calls = []

    def enroll(self, path):
        return {"voice": path}

    def synthesize(self, text, target_se):
        self.calls.append(text)
        return np.zeros(8, dtype=np.float32), 24000


def session_with(cloner, monkeypatch):
    monkeypatch.setattr(engine_factory, "create_cloner", lambda *a, **k: cloner)
    monkeypatch.setattr("voice_clone_studio.pipeline.create_cloner", lambda *a, **k: cloner)
    s = VoiceCloneSession(Engine.PORTABLE, model="chatterbox")
    s.enroll("ref.wav")
    return s


def test_a_style_is_refused_rather_than_ignored(monkeypatch):
    """Dropping it silently would look like the model ignoring what was
    asked for -- the caller gets a flat reading and no explanation."""
    session = session_with(FakeCloner(), monkeypatch)
    with pytest.raises(ValueError, match="no delivery styles"):
        session.synthesize("hello", style="angry")


def test_the_default_style_passes_straight_through(monkeypatch):
    cloner = FakeCloner()
    session = session_with(cloner, monkeypatch)
    audio, rate = session.synthesize("hello")
    assert rate == 24000 and cloner.calls == ["hello"]


# --- splitting long text ----------------------------------------------------


def test_one_sentence_stays_one_chunk():
    assert chatterbox_model.split_sentences("Just the one sentence here.") == ["Just the one sentence here."]


def test_sentences_are_split_so_no_pass_runs_past_the_token_budget():
    """A single generation is capped at ~30s of speech, so a paragraph
    handed over whole would stop partway through."""
    chunks = chatterbox_model.split_sentences(
        "This is the first sentence of the paragraph. And this is the second one, also long enough. "
        "Then a third arrives to finish the thought."
    )
    assert len(chunks) == 3
    assert chunks[0].startswith("This is the first")


def test_a_short_fragment_is_merged_rather_than_spoken_alone():
    """Very short generations lose the speaker's rhythm, so they ride
    along with their neighbour."""
    chunks = chatterbox_model.split_sentences("A long enough opening sentence to stand alone. Yes. Another full one here.")
    assert "Yes." in chunks[0]


def test_an_over_long_sentence_is_cut_at_a_comma():
    clause = "a clause that keeps going and going without ever stopping"
    text = ", ".join([clause] * 8) + "."
    chunks = chatterbox_model.split_sentences(text, limit=120)
    assert len(chunks) > 1
    assert all(len(c) <= 140 for c in chunks)
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")


def test_a_paralinguistic_tag_does_not_start_a_new_chunk():
    chunks = chatterbox_model.split_sentences("That is hilarious [laugh] and I mean it.")
    assert chunks == ["That is hilarious [laugh] and I mean it."]


def test_empty_text_still_returns_something_speakable():
    assert chatterbox_model.split_sentences("   ") == [""]


# --- sampling ---------------------------------------------------------------


def test_already_used_tokens_are_penalized():
    """Without this the model can lock onto one sound and repeat it for
    the whole generation budget."""
    logits = np.array([[2.0, -2.0, 1.0]], dtype=np.float32)
    generated = np.array([[0, 1]], dtype=np.int64)
    out = chatterbox_model._penalize(logits, generated, penalty=2.0)
    assert out[0, 0] == pytest.approx(1.0)   # positive score divided down
    assert out[0, 1] == pytest.approx(-4.0)  # negative score pushed further down
    assert out[0, 2] == pytest.approx(1.0)   # untouched


# --- file naming ------------------------------------------------------------


def test_the_quantized_variant_names_match_the_repo_layout():
    """q4f16 is the one that is fast on a CPU: 536MB at 2.5x realtime,
    against fp16's 1584MB at 14.1x."""
    assert chatterbox_model.DTYPE == "q4f16"
    files = chatterbox_model._graph_files()
    assert files["language_model"] == "onnx/language_model_q4f16.onnx"
    assert chatterbox_model._graph_files("fp32")["language_model"] == "onnx/language_model.onnx"
    assert chatterbox_model._graph_files("q8")["language_model"] == "onnx/language_model_quantized.onnx"
    assert set(files) == set(chatterbox_model.GRAPHS)
