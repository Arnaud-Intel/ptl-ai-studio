"""Document Q&A answers from what the folder holds now.

Reported on 2026-10-09: "it told him it had looked at three files when only
two were given, and made some things up". The index of a folder was kept by
the folder's path alone, so a folder indexed once went on answering from the
files it had then -- a file taken out since was still quoted. These tests
keep an index tied to its contents, with stand-ins for the two models."""
from __future__ import annotations

import os

import pytest

from doc_qa import pipeline
from doc_qa.pipeline import DocQASession
from doc_qa.store import VectorStore
from pantherlake_ai_core.engine import Engine


class Embeds:
    """One number per text, and a count of what it was asked to embed."""

    def __init__(self):
        self.documents = 0

    def embed_documents(self, texts):
        self.documents += len(texts)
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0]


class Answers:
    last_stats = None

    def __init__(self):
        self.prompts = []

    def prompt_budget(self, max_tokens):
        return 4000

    def count_tokens(self, text):
        return len(text.split())

    def answer(self, system, prompt, **kwargs):
        self.prompts.append(prompt)
        return "Seen."


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "create_embedder", lambda *args, **kwargs: Embeds())
    monkeypatch.setattr(pipeline, "create_llm", lambda *args, **kwargs: Answers())
    monkeypatch.setattr(pipeline, "cache_dir_for", lambda folder, key: tmp_path / "cache" / folder.name)
    return DocQASession(Engine.OPENVINO, device="CPU")


def _folder(tmp_path, **files):
    folder = tmp_path / "docs"
    folder.mkdir(exist_ok=True)
    for name, text in files.items():
        (folder / name).write_text(text, encoding="utf-8")
    return folder


def test_a_file_taken_out_of_the_folder_is_no_longer_answered_from(tmp_path, session):
    folder = _folder(tmp_path, **{"policy.md": "Hotels up to 180 EUR.", "trip.txt": "Alex went to Lyon.", "old.md": "The pilot was cancelled."})
    assert session.ingest(folder) == 3 and session.store.sources == ["old.md", "policy.md", "trip.txt"]
    (folder / "old.md").unlink()
    assert session.ingest(folder) == 2 and session.store.sources == ["policy.md", "trip.txt"]
    session.ask("Was the pilot cancelled?")
    assert "The pilot was cancelled." not in session.llm.prompts[-1] and "old.md" not in session.llm.prompts[-1]


def test_a_file_added_or_rewritten_is_read_and_an_untouched_folder_is_not_read_again(tmp_path, session):
    folder = _folder(tmp_path, **{"policy.md": "Hotels up to 180 EUR."})
    session.ingest(folder)
    assert session.embedder.documents == 1
    session.ingest(folder)
    assert session.embedder.documents == 1  # nothing changed: the kept index is used, as before
    (folder / "trip.txt").write_text("Alex went to Lyon.", encoding="utf-8")
    assert session.ingest(folder) == 2 and session.embedder.documents == 3
    rewritten = folder / "policy.md"
    rewritten.write_text("Hotels up to 220 EUR.", encoding="utf-8")
    os.utime(rewritten, ns=(rewritten.stat().st_atime_ns, rewritten.stat().st_mtime_ns + 2_000_000_000))
    session.ingest(folder)
    assert any("220" in chunk.text for chunk in session.store.chunks) and not any("180" in chunk.text for chunk in session.store.chunks)
    # A file that is not a document does not make the folder another folder.
    (folder / "notes.docx").write_bytes(b"not read")
    before = session.embedder.documents
    session.ingest(folder)
    assert session.embedder.documents == before


def test_an_index_kept_before_its_contents_were_recorded_is_built_again(tmp_path, session):
    folder = _folder(tmp_path, **{"policy.md": "Hotels up to 180 EUR."})
    session.ingest(folder)
    (tmp_path / "cache" / "docs" / "built-from.json").unlink()  # as every index saved by an earlier version is
    session.ingest(folder)
    assert session.embedder.documents == 2
    # The store's other user (smart-recall) keeps and loads without a fingerprint, as it always has.
    assert VectorStore.load(tmp_path / "cache" / "docs").size == 1


def test_an_indexing_that_fails_does_not_leave_the_last_folder_answering(tmp_path, session):
    session.ingest(_folder(tmp_path, **{"policy.md": "Hotels up to 180 EUR."}))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        session.ingest(empty)
    assert session.store.size == 0 and session.folder is None
    with pytest.raises(RuntimeError):
        session.ask("What is the hotel limit?")


def test_an_index_made_by_another_reading_of_the_embedding_model_is_not_this_ones(tmp_path, monkeypatch):
    keys = []
    monkeypatch.setattr(pipeline, "create_llm", lambda *args, **kwargs: Answers())
    monkeypatch.setattr(pipeline, "cache_dir_for", lambda folder, key: keys.append(key) or tmp_path / "cache" / key.replace(":", "_").replace("/", "_"))
    folder = _folder(tmp_path, **{"policy.md": "Hotels up to 180 EUR."})
    for identity in ("qwen3-embedding-0.6b/first-token", "qwen3-embedding-0.6b/last-token"):
        embedder = Embeds()
        embedder.identity = identity
        monkeypatch.setattr(pipeline, "create_embedder", lambda *args, **kwargs: embedder)
        session = DocQASession(Engine.OPENVINO, device="CPU")
        session.ingest(folder)
        assert embedder.documents == 1  # built for this reading, not taken from the other's
    assert keys[0] != keys[1] and "last-token" in keys[1]


def test_excerpts_are_named_by_their_file_and_the_answer_is_asked_for_in_sentences(tmp_path, session):
    session.ingest(_folder(tmp_path, **{"policy.md": "Hotels up to 180 EUR.", "trip.txt": "Alex went to Lyon."}))
    answer = session.ask("What is the hotel limit?")
    prompt = session.llm.prompts[-1]
    assert "Excerpt from the file policy.md:\nHotels up to 180 EUR." in prompt and prompt.endswith("Question: What is the hotel limit?")
    # No numbers to cite: "[3]" in an answer read as a third file to somebody who had given two,
    # and a prompt that ended on "refer to the excerpts by their [number]" got "[1]" for a whole answer.
    assert "[1]" not in prompt and "[number]" not in pipeline._SYSTEM_PROMPT
    assert "full sentences" in pipeline._SYSTEM_PROMPT and "do not say" in pipeline._SYSTEM_PROMPT
    assert sorted({hit.chunk.source for hit in answer.sources}) == ["policy.md", "trip.txt"]


def test_on_the_npu_texts_are_embedded_one_at_a_time_and_elsewhere_together():
    from doc_qa.embedder_openvino import OpenVINOEmbedder

    class Pipeline:
        """As the NPU's graph of the model: a call with two texts fails."""

        def __init__(self, batch_fails):
            self.calls, self.batch_fails = [], batch_fails

        def embed_documents(self, texts):
            self.calls.append(len(texts))
            if self.batch_fails and len(texts) > 1:
                raise RuntimeError("Check 'padded_shape[i] == position_shape[i]' failed")
            return [[float(len(text))] for text in texts]

    def embedder(device):
        made = OpenVINOEmbedder.__new__(OpenVINOEmbedder)
        made.device, made._one_at_a_time, made.pipeline = device, device.startswith("NPU"), Pipeline(device.startswith("NPU"))
        return made

    on_npu, on_gpu = embedder("NPU"), embedder("GPU.0")
    assert on_npu.embed_documents(["a", "bb", "ccc"]) == [[1.0], [2.0], [3.0]] and on_npu.pipeline.calls == [1, 1, 1]
    assert on_gpu.embed_documents(["a", "bb", "ccc"]) == [[1.0], [2.0], [3.0]] and on_gpu.pipeline.calls == [3]


def test_a_question_nothing_in_the_folder_is_close_to_is_not_given_to_the_model(tmp_path, monkeypatch):
    class Reads:
        """Texts about hotels point one way, everything else another; and it knows how its scores read."""

        relevance_floor = 0.30

        def embed_documents(self, texts):
            return [[1.0, 0.0] if "hotel" in text.lower() else [0.0, 1.0] for text in texts]

        def embed_query(self, text):
            return [1.0, 0.1] if "hotel" in text.lower() else [0.2, -1.0] if "pilot" in text.lower() else [0.1, 1.0]

    monkeypatch.setattr(pipeline, "create_embedder", lambda *args, **kwargs: Reads())
    monkeypatch.setattr(pipeline, "create_llm", lambda *args, **kwargs: Answers())
    monkeypatch.setattr(pipeline, "cache_dir_for", lambda folder, key: tmp_path / "cache")
    session = DocQASession(Engine.OPENVINO, device="CPU")
    session.ingest(_folder(tmp_path, **{"policy.md": "Hotel nights up to 180 EUR.", "trip.txt": "Alex went to Lyon."}))

    nothing = session.ask("What budget was approved for the pilot?")
    assert nothing.text == pipeline.NOTHING_CLOSE and nothing.sources == [] and session.llm.prompts == []
    # A question a passage is close to is answered from that passage, without the ones beside the point.
    answer = session.ask("What is the hotel limit?")
    assert answer.text == "Seen." and [hit.chunk.source for hit in answer.sources] == ["policy.md"]
    assert "Alex went to Lyon." not in session.llm.prompts[-1]


def test_asked_alone_the_model_is_given_the_question_and_nothing_else(tmp_path, session):
    """What the documents change is shown by asking without them: the same
    model, the same question, no excerpt -- and no index needed for it."""
    told = []
    session.llm.answer = lambda system, prompt, **kwargs: told.append((system, prompt)) or "I do not have that information."
    answer = session.ask_alone("Who decides on the Lyon pilot?")
    assert told == [(pipeline.ALONE_PROMPT, "Who decides on the Lyon pilot?")]  # the question as asked, and nothing to read
    assert answer.text == "I do not have that information." and answer.sources == []
    # Asked with the documents afterwards, it is given the passages as before: one does not change the other.
    session.ingest(_folder(tmp_path, **{"decision.md": "Priya Desai decides on September 17."}))
    session.ask("Who decides on the Lyon pilot?")
    assert "Priya Desai decides" in told[-1][1] and told[-1][0] != pipeline.ALONE_PROMPT


def test_the_launcher_asks_alone_without_a_folder_and_says_which_way_it_answered(tmp_path, session, monkeypatch):
    from fastapi.testclient import TestClient

    from launcher import app as launcher_app
    from launcher.doc_qa_runner import DocQARunner

    monkeypatch.setattr(pipeline, "DocQASession", lambda *args, **kwargs: session, raising=False)
    import launcher.doc_qa_runner as runner_module

    monkeypatch.setattr(runner_module, "DocQASession", lambda *args, **kwargs: session)
    monkeypatch.setattr(launcher_app, "doc_qa_runner", DocQARunner())
    monkeypatch.setattr(launcher_app, "resolve", lambda engine, device, **kwargs: (Engine.OPENVINO, device or "CPU"))
    web = TestClient(launcher_app.app)
    # Nothing loaded, and nothing said about what to load: there is nobody to ask.
    assert web.post("/api/doc-qa/ask", json={"question": "Who decides?", "alone": True}).status_code == 409
    assert web.post("/api/doc-qa/ask", json={"question": "  ", "alone": True}).status_code == 400
    alone = web.post("/api/doc-qa/ask", json={"question": "Who decides?", "alone": True, "engine": "openvino", "compute_device": "NPU"})
    assert alone.status_code == 200 and alone.json()["alone"] is True and alone.json()["sources"] == []
    assert session.llm.prompts == ["Who decides?"] and web.get("/api/doc-qa/status").json()["indexed"] is False
    # With the documents, a folder is still needed first.
    refused = web.post("/api/doc-qa/ask", json={"question": "Who decides?"})
    assert refused.status_code == 409 and "Index a folder first" in refused.json()["error"]
    folder = _folder(tmp_path, **{"decision.md": "Priya Desai decides on September 17."})
    assert web.post("/api/doc-qa/ingest", json={"folder": str(folder), "engine": "openvino", "compute_device": "NPU"}).status_code == 200
    answered = web.post("/api/doc-qa/ask", json={"question": "Who decides?"}).json()
    assert answered["alone"] is False and [source["source"] for source in answered["sources"]] == ["decision.md"]
