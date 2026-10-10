"""The small language model a brick answers with can be chosen (BACKLOG R36):
Qwen2.5-1.5B, the fastest, or Qwen3-8B. Which one each brick takes when
nobody chose comes from measurements written beside each default; what is
checked here is that the choice reaches the model that is loaded, and that
a larger model is never fetched just because it is preferred."""
from __future__ import annotations

import pytest
from doc_qa import language_models
from doc_qa import pipeline as doc_qa
from pantherlake_ai_core.engine import Engine

QUICK, CAREFUL = language_models.QUICK.key, language_models.CAREFUL.key


def test_a_model_comes_in_one_build_for_the_npu_and_one_for_the_rest():
    on_npu = language_models.repo_for(CAREFUL, Engine.OPENVINO, "NPU")
    elsewhere = language_models.repo_for(CAREFUL, Engine.OPENVINO, "GPU.0")
    assert on_npu and elsewhere and on_npu != elsewhere  # the standard build does not compile for the NPU
    assert elsewhere == language_models.repo_for(CAREFUL, Engine.OPENVINO, "CPU")
    for device in ("NPU", "GPU", "CPU"):
        assert language_models.repo_for(QUICK, Engine.OPENVINO, device) is None  # the backend's own default
        assert language_models.repo_for(None, Engine.OPENVINO, device) is None
    assert language_models.repo_for(QUICK, Engine.PORTABLE, "cpu") is None


def test_a_model_nobody_has_or_that_the_engine_lacks_is_refused_by_name():
    with pytest.raises(ValueError, match="Unknown language model 'gpt-9'"):
        language_models.get("gpt-9")
    with pytest.raises(ValueError, match="needs the OpenVINO engine"):
        language_models.repo_for(CAREFUL, Engine.PORTABLE, "cpu")
    assert [model.key for model in language_models.choices(Engine.PORTABLE)] == [QUICK]
    assert [model.key for model in language_models.choices(Engine.OPENVINO)] == [QUICK, CAREFUL]


def test_a_preferred_model_is_taken_only_where_the_laptop_has_it(monkeypatch):
    """A better answer is not worth gigabytes fetched because somebody
    pressed Ask in front of an audience."""
    from pantherlake_ai_core import model_cache

    on_disk = {language_models.CAREFUL.repo_npu}
    monkeypatch.setattr(model_cache, "is_repo_cached", lambda repo: repo in on_disk)
    assert language_models.preferred(CAREFUL, Engine.OPENVINO, "NPU") == CAREFUL
    assert language_models.preferred(CAREFUL, Engine.OPENVINO, "GPU.0") == QUICK  # that build is not there
    assert language_models.preferred(CAREFUL, Engine.PORTABLE, "cpu") == QUICK  # nor is it for this engine
    assert language_models.preferred(QUICK, Engine.OPENVINO, "NPU") == QUICK
    assert language_models.on_disk(QUICK, Engine.OPENVINO, "GPU.0")  # what every installation fetches first


class _Model:
    last_stats = None


@pytest.fixture
def loads(monkeypatch):
    """Every language model the bricks ask the factory for, as (device, repository)."""
    asked = []

    def create_llm(engine, *, device="AUTO", model_repo=None, **kwargs):
        asked.append((device, model_repo))
        return _Model()

    monkeypatch.setattr(doc_qa, "create_llm", create_llm)
    monkeypatch.setattr(doc_qa, "create_embedder", lambda *args, **kwargs: object())
    return asked


def test_document_qa_prefers_the_larger_model_and_falls_back_on_the_small_one(loads, monkeypatch):
    assert doc_qa.PREFERRED_MODEL == CAREFUL
    monkeypatch.setattr(language_models, "on_disk", lambda key, engine, device: True)
    session = doc_qa.DocQASession(Engine.OPENVINO, device="NPU")
    assert session.model == CAREFUL and loads == [("NPU", language_models.CAREFUL.repo_npu)]

    monkeypatch.setattr(language_models, "on_disk", lambda key, engine, device: key == QUICK)
    session = doc_qa.DocQASession(Engine.OPENVINO, device="NPU")
    assert session.model == QUICK and loads[-1] == ("NPU", None)
    # Named, it is loaded whether or not it is there: that is what a first use downloads.
    assert doc_qa.DocQASession(Engine.OPENVINO, device="GPU.0", model=CAREFUL).model == CAREFUL
    assert loads[-1] == ("GPU.0", language_models.CAREFUL.repo)
    with pytest.raises(ValueError, match="needs the OpenVINO engine"):
        doc_qa.DocQASession(Engine.PORTABLE, device="cpu", model=CAREFUL)


def test_the_model_can_change_between_two_questions_and_the_index_stays(loads):
    session = doc_qa.DocQASession(Engine.OPENVINO, device="NPU", model=QUICK)
    session.store, session.folder = "the index", "the folder"
    embedder = session.embedder
    assert session.use_model(QUICK) is False and len(loads) == 1  # already the one
    assert session.use_model(CAREFUL) is True and session.model == CAREFUL
    assert loads[-1] == ("NPU", language_models.CAREFUL.repo_npu)
    assert (session.store, session.folder, session.embedder) == ("the index", "the folder", embedder)


def test_the_other_bricks_keep_the_small_model_unless_told():
    """Measured on the NPU, 2026-10-10: the 8B gets more amounts wrong on the
    sample receipts, invents a meeting when asked to recall one, and says a
    comment three times slower with less of a voice."""
    from expense_extract import pipeline as expenses
    from video_commentary import pipeline as commentary
    from voice_assistant import session as voice

    assert expenses.DEFAULT_MODEL == voice.DEFAULT_MODEL == commentary.DEFAULT_MOOD_MODEL == QUICK


def test_the_commentator_tells_each_model_what_it_needs_to_hear():
    """The 8B obeys "add none" by giving the sentence back as it was (29
    lines of 48): it is told to use new words. The 1.5B needs holding back."""
    from video_commentary import moods

    upbeat = moods.get("upbeat").instruction
    assert moods.worded_for(upbeat, QUICK) == upbeat and "add none" in upbeat
    for_the_8b = moods.worded_for(upbeat, CAREFUL)
    assert for_the_8b != upbeat and "do not repeat the sentence" in for_the_8b and "add no detail" in for_the_8b
    assert for_the_8b.startswith(upbeat.split(" One short sentence")[0])  # the mood itself is the same


def test_the_commentator_loads_the_model_it_was_asked_for(monkeypatch):
    import threading

    from doc_qa import engine_factory
    from video_commentary import moods, pipeline

    class Says:
        last_stats = None

        def __init__(self):
            self.told = []

        def answer(self, instruction, line, **kwargs):
            self.told.append(instruction)
            return "A line."

    asked, model = [], Says()
    monkeypatch.setattr(engine_factory, "create_llm", lambda engine, *, device, model_repo=None, **kw: asked.append((device, model_repo)) or model)

    class Eyes:
        def __init__(self, **kwargs):
            pass

        def ask(self, picture, question, **kwargs):
            return "A cowboy herds cattle.", None

    import numpy as np
    from screen_ocr import extractor_openvino

    monkeypatch.setattr(extractor_openvino, "OpenVINOExtractor", Eyes)
    stop, said = threading.Event(), []

    def frames(stopped):
        while not stopped.is_set():
            yield np.zeros((48, 64, 3), dtype=np.uint8)

    def on_comment(comment, speech):
        said.append(comment)
        stop.set()

    pipeline.run(source="file", vision_device="GPU", mood_device="NPU", mood=lambda: "sports", mood_model=CAREFUL,
                 frames=frames, on_comment=on_comment, stop_event=stop, every=0.1)
    assert asked == [("NPU", language_models.CAREFUL.repo_npu)]
    assert model.told == [moods.worded_for(moods.get("sports").instruction, CAREFUL)]
    with pytest.raises(ValueError, match="Unknown language model"):
        pipeline.run(source="file", vision_device="GPU", mood_device="NPU", mood=lambda: "sports", mood_model="gpt-9", frames=frames)
