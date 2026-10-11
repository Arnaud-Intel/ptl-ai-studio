"""A page a brick has made, kept so that parts of it can be changed.

HTML Creator and the Page Agent both end with a page, and both can now be
asked to write one part of it again (`html_creator.revise`): the part a
person pointed at in the page, with a comment. This is what the two runners
share: the page as it was made, each version it has been through since, the
pictures it may use, and the one way a batch of changes is applied to it.

A version is the page as it was written: its pictures by their file names.
What a browser is shown is made from it when asked for -- the page with its
pictures put in, and the copy of it in which every part carries a number,
which is the one that is pointed at and never the one that is kept.
"""
from __future__ import annotations

import threading
from typing import Callable

from html_creator import pictures as picture_kit
from html_creator import revise
from html_creator.types import Revised

from .errors import Conflict

# write(source, change, pictures) -> Revised: the models' side of a change,
# which each runner does with its own session. `change` is the change as it
# was asked: its part, its comment, what the browser saw inside the part --
# and, where the comment is about a picture, that picture's file name.
Write = Callable[[str, dict, "list[picture_kit.Picture]"], Revised]


class PageRevisions:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._versions: list[str] = []
        # What made each version out of the one before it: nothing, for the first.
        self._changes: list[list[dict]] = []
        self._pictures: list[picture_kit.Picture] = []

    # ------------------------------------------------------------------ the page

    def start(self, source: str, pictures: list[picture_kit.Picture] | None = None) -> None:
        """A page has just been made: it is the first version, and the
        versions of the page before it are gone."""
        with self._lock:
            self._versions = [source] if source and source.strip() else []
            self._changes = [[]] if self._versions else []
            self._pictures = list(pictures or [])

    def forget(self) -> None:
        self.start("")

    @property
    def available(self) -> bool:
        return bool(self._versions)

    @property
    def version(self) -> int:
        """The number of the version in hand: 0 for the page as it was made."""
        return len(self._versions) - 1

    @property
    def source(self) -> str:
        if not self._versions:
            raise Conflict("There is no page to change yet: make one first.")
        return self._versions[-1]

    @property
    def pictures(self) -> list[picture_kit.Picture]:
        return list(self._pictures)

    def add_picture(self, picture: picture_kit.Picture) -> None:
        """One more file the page may use (a picture drawn again)."""
        with self._lock:
            self._pictures = [kept for kept in self._pictures if kept.name != picture.name] + [picture]

    def undo(self) -> None:
        with self._lock:
            if len(self._versions) < 2:
                raise Conflict("This is the page as it was made: there is nothing to undo.")
            self._versions.pop()
            self._changes.pop()

    def state(self, first: bool = False) -> dict:
        """What the page shows: the version in hand, with its pictures in it
        (`html`), the copy of it to point at (`marked`), and how it came to
        be what it is. `first`: the page as it was made as well
        (`first_html`), for a browser that has lost it."""
        with self._lock:
            versions, changes, pictures = list(self._versions), [list(batch) for batch in self._changes], list(self._pictures)
        if not versions:
            return {"available": False, "version": 0, "versions": 0, "html": "", "marked": "", "history": [], "can_undo": False}
        source = versions[-1]
        return {
            "available": True,
            "version": len(versions) - 1,
            "versions": len(versions),
            "html": picture_kit.embed(source, pictures)[0],
            "marked": picture_kit.embed(revise.marked(source), pictures)[0],
            "mark": revise.MARK,
            # As written, for whoever wants to read it: with its pictures in, a page is megabytes.
            "html_source": source,
            "history": [{"version": index, "changes": batch} for index, batch in enumerate(changes) if index],
            "can_undo": len(versions) > 1,
            **({"first_html": picture_kit.embed(versions[0], pictures)[0]} if first else {}),
        }

    # ------------------------------------------------------------------ changing it

    def check(self, version: int, changes: list[dict]) -> list[dict]:
        """`changes` as they will be made, or why they cannot be: each is
        `{"part": number, "comment": text, "seen": numbers}`. Refused here,
        before a model is asked anything: a page that has moved on since it
        was pointed at, a part the page does not have, a part with nothing
        said about it, two comments of which one is inside the other."""
        if version != self.version:
            raise Conflict("The page has changed since these parts were pointed at: point at them again on the page as it is now.")
        if not changes:
            raise ValueError("Point at a part of the page and say what to change.")
        source = self.source
        numbers = [int(change["part"]) for change in changes]
        if len(set(numbers)) != len(numbers):
            raise ValueError("The same part is commented twice: say it in one comment.")
        for change in changes:
            if not str(change.get("comment") or "").strip():
                raise ValueError(f"Part {change['part']} has no comment: say what to change, or take it off the list.")
            revise.find(source, int(change["part"]))  # a ValueError if the page has no such part
        for outer in numbers:
            within = set(revise.inside(source, outer))
            nested = sorted(within & set(numbers))
            if nested:
                raise ValueError(
                    f"Part {nested[0]} is inside part {outer}: comment one of them, apply it, then the other "
                    "(or say both things in the comment on the larger part)."
                )
        return changes

    def change(
        self, version: int, changes: list[dict], write: Write,
        on_start: Callable[[int, int, dict], None] = lambda index, total, change: None,
        stopped: Callable[[], bool] = lambda: False,
    ) -> list[dict]:
        """Make `changes` to version `version`, one after the other, and
        keep the page they leave as the next version -- if at least one of
        them could be made. Returns what became of each, in the order given.

        From the last part of the page to the first: the parts are numbered
        in the order they come, so a change further down leaves the numbers
        of everything before it as they were pointed at. Pictures to draw
        again come after every part to write again: a picture changes no
        number, and on a machine where the two models share a chip they
        then change places once and not at every turn."""
        changes = self.check(version, changes)
        source = self.source
        results: dict[int, dict] = {}
        order = sorted(range(len(changes)), key=lambda index: (bool(changes[index].get("picture")), -int(changes[index]["part"])))
        for done, index in enumerate(order, 1):
            change = changes[index]
            if stopped():
                results[index] = {"part": int(change["part"]), "comment": change["comment"].strip(), "done": False,
                                  "problem": "stopped before this part was reached"}
                continue
            on_start(done, len(order), change)
            revised = write(source, change, self.pictures)
            stats = revised.stats
            results[index] = {
                "part": revised.number, "comment": revised.comment, "done": not revised.problem, "problem": revised.problem,
                "plan": revised.plan, "notes": revised.notes, "attempts": revised.attempts, "seconds": revised.seconds,
                "tokens": getattr(stats, "tokens", None), "tokens_per_second": getattr(stats, "tokens_per_second", None),
                "rules_added": bool(revised.css), "script_added": bool(revised.script),
                "picture": change.get("picture") or None,
                "characters": [len(revised.before), len(revised.after)],
            }
            if not revised.problem:
                source = revised.source
        made = [results[index] for index in range(len(changes))]
        if any(result["done"] for result in made):
            with self._lock:
                self._versions.append(source)
                self._changes.append(made)
        return made
