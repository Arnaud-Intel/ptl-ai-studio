"""Live translation's transcript is kept by the launcher and can be handed to
Meeting Notes for a summary: one recording, two uses."""
from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient
from meeting_notes.session import MeetingSession
from meeting_notes.types import TranscriptLine
from pantherlake_ai_core.engine import Engine
from pantherlake_ai_core.types import TranslationResult

from launcher import app as launcher_app
from launcher import live_translation_runner as live_module
from launcher.errors import Conflict
from launcher.live_translation_runner import LiveTranslationRunner
from launcher.meeting_notes_runner import MeetingNotesRunner


def _session(runner: LiveTranslationRunner, monkeypatch, texts: list[str]) -> list[dict]:
    """One Start-to-Stop of live translation that hears `texts`; returns what
    was sent to the page."""

    def fake_run(**kwargs):
        kwargs["on_ready"]()
        for text in texts:
            kwargs["on_result"](TranslationResult(text=text, detected_language="fr", language_probability=1.0))

    monkeypatch.setattr(live_module.pipeline, "run", fake_run)

    async def run() -> list[dict]:
        queue: asyncio.Queue = asyncio.Queue()
        runner.start(
            loop=asyncio.get_running_loop(), queue=queue, source="mic", audio_device=None,
            engine=Engine.OPENVINO, model_size="base", compute_device="CPU",
        )
        sent = []
        while not sent or sent[-1]["type"] != "stopped":
            sent.append(await asyncio.wait_for(queue.get(), 5))
        await asyncio.to_thread(runner._thread.join, 5)
        return [message for message in sent if message["type"] == "result"]

    return asyncio.run(run())


@pytest.fixture
def quiet_events(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher_app.events, "LOG_FILE", tmp_path / "events.log")


def test_the_transcript_carries_on_across_stop_and_start(quiet_events, monkeypatch):
    runner = LiveTranslationRunner()
    first = _session(runner, monkeypatch, ["good morning", "let us start"])
    # A restart mid-meeting (to change the spoken language, say) is the same meeting.
    second = _session(runner, monkeypatch, ["one more thing"])

    kept = runner.transcript()
    assert [(line["seq"], line["text"], line["detected_language"]) for line in kept["lines"]] == [
        (1, "good morning", "fr"), (2, "let us start", "fr"), (3, "one more thing", "fr"),
    ]
    # Each line reaches the page named, so one drawn from transcript() is not drawn twice.
    assert [(m["transcript"], m["seq"]) for m in first + second] == [(kept["id"], 1), (kept["id"], 2), (kept["id"], 3)]
    assert first[0]["timestamp"] == kept["lines"][0]["timestamp"]


def test_clearing_starts_a_new_transcript_with_an_id_never_used_before(quiet_events, monkeypatch):
    runner = LiveTranslationRunner()
    _session(runner, monkeypatch, ["the first meeting"])
    old = runner.transcript()["id"]

    assert runner.clear_transcript() > old and runner.clear_transcript() > old + 1  # twice in one millisecond too
    assert runner.transcript()["lines"] == []
    assert _session(runner, monkeypatch, ["the second meeting"])[0]["seq"] == 1


def test_a_session_given_a_transcript_can_write_notes_at_once():
    lines = [TranscriptLine("10:00:00", "we agreed to ship on friday", "en")]
    session = MeetingSession(Engine.OPENVINO, compute_device="GPU.0", whisper_model_size="base", transcript=lines)
    assert session.transcript_text() == "[10:00:00] we agreed to ship on friday"

    session._llm = model = object()
    session.replace_transcript([TranscriptLine("11:00:00", "another meeting", "en")])
    assert session.transcript_text() == "[11:00:00] another meeting" and session._llm is model  # the model stays loaded


class _NotesLLM:
    last_stats = None

    def __init__(self):
        self.read: list[str] = []

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def prompt_budget(self, max_tokens: int) -> int:
        return 4000

    def answer(self, system_prompt: str, user_prompt: str, max_tokens: int = 512, control=None) -> str:
        self.read.append(user_prompt)
        return "- Summary\nAction items\n- Maria: send the plan (Friday)"


_HEARD = [
    "good morning everyone thank you for joining the weekly review",
    "maria will send the launch plan to the whole team by friday",
    "we meet again on monday to go through what is still open",
]


@pytest.fixture
def studio(quiet_events, monkeypatch):
    """The launcher with fresh runners, a live translation transcript of
    `_HEARD`, and a notes model that records what it was given to read."""
    live, meeting, llm = LiveTranslationRunner(), MeetingNotesRunner(), _NotesLLM()
    _session(live, monkeypatch, _HEARD)
    monkeypatch.setattr(launcher_app, "live_translation_runner", live)
    monkeypatch.setattr(launcher_app, "meeting_notes_runner", meeting)
    monkeypatch.setattr(launcher_app, "resolve", lambda engine, device, **kwargs: (Engine.OPENVINO, device or "GPU.0"))
    monkeypatch.setattr("meeting_notes.session.create_llm", lambda engine, **kwargs: llm)
    return TestClient(launcher_app.app), live, meeting, llm


def test_the_page_can_read_the_transcript_back_and_clear_it(studio):
    client, live, _, _ = studio
    read = client.get("/api/live-translation/transcript").json()
    assert read["id"] == live.transcript()["id"] and [line["text"] for line in read["lines"]] == _HEARD

    cleared = client.delete("/api/live-translation/transcript").json()
    assert cleared["id"] > read["id"]
    assert client.get("/api/live-translation/transcript").json() == {"id": cleared["id"], "lines": []}


def test_meeting_notes_summarises_what_live_translation_heard(studio):
    client, live, meeting, llm = studio
    handed = client.post("/api/meeting-notes/from-live-translation", json={"engine": "openvino", "compute_device": "NPU"})
    assert handed.status_code == 200
    assert [line["text"] for line in handed.json()["lines"]] == _HEARD
    assert handed.json()["words"] == sum(len(text.split()) for text in _HEARD)
    assert meeting._session.compute_device == "NPU"  # the notes are written where Meeting Notes was set to

    notes = client.post("/api/meeting-notes/generate").json()
    assert notes["transcript_line_count"] == 3 and "Maria: send the plan" in notes["text"]
    assert all(text in llm.read[0] for text in _HEARD)  # the whole transcript, not the last line

    # Live translation heard more since: handing over again takes all of it,
    # and the notes model loaded for the first summary is not loaded again.
    session = meeting._session
    with live._transcript_lock:
        live._transcript.append({"seq": 4, "timestamp": "10:05:00", "text": "and thank you all", "detected_language": "en"})
    again = client.post("/api/meeting-notes/from-live-translation", json={"engine": "openvino", "compute_device": "NPU"})
    assert len(again.json()["lines"]) == 4 and meeting._session is session
    assert client.post("/api/meeting-notes/generate").json()["transcript_line_count"] == 4


def test_nothing_heard_is_nothing_to_summarise(studio):
    client, _, meeting, _ = studio
    client.delete("/api/live-translation/transcript")
    refused = client.post("/api/meeting-notes/from-live-translation", json={})
    assert refused.status_code == 409 and "Nothing has been transcribed yet" in refused.json()["error"]
    assert meeting._session is None


def test_a_meeting_being_transcribed_is_not_replaced(studio, monkeypatch):
    client, _, meeting, _ = studio
    monkeypatch.setattr(MeetingNotesRunner, "running", property(lambda self: True))
    refused = client.post("/api/meeting-notes/from-live-translation", json={})
    assert refused.status_code == 409 and "transcribing a meeting of its own" in refused.json()["error"]
    with pytest.raises(Conflict):
        meeting.adopt([{"text": "hello"}], engine=Engine.OPENVINO, compute_device="GPU.0", whisper_model_size="base")
