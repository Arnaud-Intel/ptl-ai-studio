"""The conductor: plain code on the CPU that puts three models to work on
three chips and checks what comes back. No model of its own.

    request --> [plan]   a small model on the NPU names the page, says what
                         it presents and briefs six photographs
            --> [images] an image model takes them          } at the same
            --> [page]   a coding model writes the HTML     } time, with two
                         around their file names, told how  } GPUs
                         the studio builds a page
            --> [check]  the conductor puts the studio's layout rules into
                         the page, verifies it, has any picture it still
                         lacks drawn and, for a fault a second try can
                         fix, asks for the page once more

The page is written from the pictures' descriptions, never from the pictures
themselves, which is why the two middle steps can overlap: with a discrete
GPU beside the integrated one, each model has its own chip. With one GPU
they take turns, and each is unloaded before the other is loaded -- a 30B
coding model and an image model do not belong in the same memory.
"""
from __future__ import annotations

import gc
import threading
import time
import zlib
from pathlib import Path
from typing import Callable

from doc_qa.engine_factory import create_llm
from html_creator import pictures as picture_kit
from html_creator.session import HtmlCreatorSession
from pantherlake_ai_core import npu
from pantherlake_ai_core.engine import Engine, GpuDevice
from pantherlake_ai_core.types import GenerationControl, stopped

from . import art_direction
from . import checks as checking
from . import plan as planning
from .repeats import give_repeats_their_own
from .runaway import Watch
from .types import Assignment, Check, DrawnPicture, PageResult

# The planner: the general-purpose model meeting-notes writes with, for the
# same reasons (see that brick) -- and already on the NPU if notes were asked
# for. The NPU needs its own build; the standard one does not compile there.
_PLANNER_REPO = "OpenVINO/Qwen3-8B-int4-ov"
_PLANNER_REPO_NPU = "OpenVINO/Qwen3-8B-int4-cw-ov"
_PLAN_MAX_TOKENS = 600  # thirteen lines, six of them a sentence describing a picture: 210 to 270 tokens
# A page built the studio's way is 5,000 to 5,500 tokens. html-creator's own
# limit (6,144) would cut one in a few off at the footer, and a page cut off
# is written again.
_PAGE_MAX_TOKENS = 8192
# The coding model's own temperature (Qwen3-Coder's generation settings), not
# the 0.2 the other bricks draw at. At 0.2 a long stylesheet now and then
# turns into a loop -- `.section-title h2 {...} .section-title p {...}` over
# and over until the tokens run out: three pages in fifteen (2026-10-08).
# At 0.7, none in thirteen, on the same requests.
_PAGE_TEMPERATURE = 0.7

PLAN, IMAGES, PAGE, CHECK = "plan", "images", "page", "check"
STEPS = (PLAN, IMAGES, PAGE, CHECK)

# on_step(step, state, device, detail): state is "loading" (a model is being
# brought up), "running", "done" or "failed".
StepReport = Callable[[str, str, str, str], None]


def planner_repo(device: str) -> str:
    return _PLANNER_REPO_NPU if npu.is_npu(device) else _PLANNER_REPO


def assign(
    devices: list[str],
    gpus: list[GpuDevice],
    *,
    planner: str | None = None,
    images: str | None = None,
    page: str | None = None,
    npu_usable: bool = True,
) -> Assignment:
    """Who does what, on a machine with `devices` (OpenVINO's names) and
    `gpus`. A chip given by name is used as given; the rest follows the
    hardware:

    - the planner goes to the NPU, else to the integrated GPU, else the CPU;
    - the coding model, the large one, goes to the discrete GPU if there is
      one, else to the integrated GPU, else the CPU;
    - the image model goes to the integrated GPU -- which leaves it alone
      there when the coding model has a discrete card, and makes the two
      share it when it has not.
    """
    integrated = [gpu.id for gpu in gpus if "dGPU" not in gpu.full_name]
    discrete = [gpu.id for gpu in gpus if "dGPU" in gpu.full_name]
    any_gpu = (integrated or discrete or [None])[0]

    if not planner:
        has_npu = npu_usable and any(npu.is_npu(device) for device in devices)
        planner = "NPU" if has_npu else (any_gpu or "CPU")
    if not page:
        page = (discrete[-1] if discrete else any_gpu) or "CPU"
    if not images:
        images = any_gpu or "CPU"
    together = images != page and images != "CPU" and page != "CPU"
    return Assignment(planner=planner, images=images, page=page, together=together)


class PageAgent:
    """Keeps the three models between requests, and decides which of them
    may stay loaded."""

    def __init__(self, engine: Engine = Engine.OPENVINO):
        if engine != Engine.OPENVINO:
            raise ValueError("The page agent needs the OpenVINO engine: there is no portable image model.")
        self.engine = engine
        self._planner = None
        self._planner_device: str | None = None
        self._images = None
        self._page: HtmlCreatorSession | None = None
        self._ran_away = False  # the page last written was stopped for going round in circles

    # ------------------------------------------------------------------ models

    def loaded(self) -> dict[str, str]:
        """Which step's model is in memory, and on which chip."""
        held = {}
        if self._planner is not None:
            held[PLAN] = getattr(self._planner, "device", self._planner_device)
        if self._images is not None:
            held[IMAGES] = self._images.device
        if self._page is not None and self._page._llm is not None:
            held[PAGE] = self._page.compute_device
        return held

    def unload(self) -> None:
        self._planner = self._images = self._page = None
        self._planner_device = None
        gc.collect()  # the runtime frees a model's memory with its last reference

    def _release_images(self) -> None:
        if self._images is not None:
            self._images = None
            gc.collect()

    def _release_page(self) -> None:
        if self._page is not None:
            self._page = None
            gc.collect()

    # ------------------------------------------------------------------- steps

    def _plan(self, request: str, device: str, report: StepReport, on_downloading, control=None):
        report(PLAN, "loading", device, "")
        started = time.perf_counter()
        try:
            if self._planner is None or self._planner_device != device:
                self._planner = None
                self._planner = create_llm(
                    self.engine, device=device, model_repo=planner_repo(device), on_downloading=on_downloading
                )
                self._planner_device = device
            # Where it really is: a planner asked onto a lost NPU carries on elsewhere.
            report(PLAN, "running", getattr(self._planner, "device", device), "")
            # The most likely word each time: nine plans out of nine were sound
            # that way, where drawing among the likely words gave one that was
            # the prompt's own example sent back and one that said the same
            # sentence for two pictures.
            answer = self._planner.answer(planning.SYSTEM_PROMPT, request, max_tokens=_PLAN_MAX_TOKENS, sample=False)
            plan = planning.parse(request, answer)
            if plan.copied and not stopped(control):
                # Asked the same way it would answer the same thing, so this time it draws.
                answer = self._planner.answer(planning.SYSTEM_PROMPT, request, max_tokens=_PLAN_MAX_TOKENS, sample=True)
                again = planning.parse(request, answer)
                plan = again if again.copied < plan.copied else plan
            stats = getattr(self._planner, "last_stats", None)
        except Exception as exc:  # a planner that cannot be asked costs the plan, not the page
            plan, stats = planning.fallback(request, f"The planner could not be used ({exc}); the page is made from the request alone."), None
        seconds = time.perf_counter() - started
        report(PLAN, "done", getattr(self._planner, "device", device) if self._planner else device, f"{len(plan.pictures)} picture(s) planned")
        return plan, stats, seconds

    def _draw(self, specs, device: str, work_dir: Path, report: StepReport, on_downloading, on_picture, control, what="picture"):
        from .images import ImageMaker

        report(IMAGES, "loading", device, "")
        started = time.perf_counter()
        if self._images is None or self._images.device != device:
            self._release_images()
            self._images = ImageMaker(device, on_downloading=on_downloading)
        drawn: list[DrawnPicture] = []
        for index, spec in enumerate(specs, 1):
            if stopped(control):
                break
            report(IMAGES, "running", device, f"{what} {index} of {len(specs)}")
            path = work_dir / spec.name
            # A seed of its own, from its name: two pictures with one
            # description differ, and the same request draws the same page.
            seed = zlib.crc32(spec.name.encode("utf-8")) % 2_000_000_000
            seconds = self._images.draw(spec.prompt, spec.width, spec.height, path, seed)
            drawn.append(DrawnPicture(spec.name, path, spec.width, spec.height, spec.prompt, round(seconds, 2)))
            if on_picture is not None:
                on_picture(drawn[-1])
        report(IMAGES, "done", device, f"{len(drawn)} picture(s) drawn")
        return drawn, time.perf_counter() - started

    def _write(
        self, prompt: str, plan, device: str, work_dir: Path, report: StepReport, on_downloading, before_embed, control,
        note: str = "",
    ):
        report(PAGE, "loading", device, "")
        started = time.perf_counter()
        if self._page is None or self._page.compute_device != device:
            self._release_page()
            self._page = HtmlCreatorSession(self.engine, compute_device=device)
        offered = [
            picture_kit.Picture(spec.name, work_dir / spec.name, "image/jpeg", spec.width, spec.height, spec.prompt)
            for spec in plan.pictures
        ]
        watch = Watch(control)  # stops a page that has started repeating itself
        result = self._page.generate(
            mode="landing_page",
            prompt=prompt,
            picture_list=offered,
            before_embed=before_embed,
            # After the pictures, where the model heeds it: what to fix from a
            # first try, then the rules about the pictures and the first line.
            closing="\n".join(part for part in (note, plan.closing()) if part),
            max_tokens=_PAGE_MAX_TOKENS,
            temperature=_PAGE_TEMPERATURE,
            on_ready=lambda: report(PAGE, "running", device, ""),
            on_downloading=on_downloading,
            control=watch.control,
        )
        self._ran_away = watch.ran_away
        if watch.ran_away:
            # Kept beside the pictures: what a page was repeating when it was
            # stopped is the first thing anybody asks, and it is gone otherwise.
            (work_dir / "stopped-page.html").write_text(result.html_source or result.html, encoding="utf-8")
        _written, shown = _finished(result, plan)
        detail = f"{len(shown)} of {len(offered)} picture(s) placed"
        report(PAGE, "done", device, "stopped: it was repeating itself" if watch.ran_away else detail)
        return result, time.perf_counter() - started

    # --------------------------------------------------------------------- run

    def build(
        self,
        request: str,
        *,
        assignment: Assignment,
        work_dir: Path,
        on_step: StepReport | None = None,
        on_plan: Callable[[planning.PagePlan], None] | None = None,
        on_picture: Callable[[DrawnPicture], None] | None = None,
        on_downloading: Callable[[], None] | None = None,
        control: GenerationControl | None = None,
    ) -> PageResult:
        """Blocks until the page is built. `on_step` follows the work step by
        step; `on_plan` and `on_picture` hand over what there is to show as
        soon as it exists; `control` receives the page as it is written and
        can stop the run."""
        if not request or not request.strip():
            raise ValueError("Describe the page to build.")
        request = request.strip()
        report: StepReport = on_step or (lambda step, state, device, detail: None)
        work_dir.mkdir(parents=True, exist_ok=True)
        began = time.perf_counter()
        seconds: dict[str, float] = {}

        plan, planner_stats, seconds[PLAN] = self._plan(request, assignment.planner, report, on_downloading, control)
        if on_plan is not None:
            on_plan(plan)
        prompt = plan.page_prompt()

        drawn: list[DrawnPicture] = []
        if assignment.together:
            # Two chips: the pictures are drawn while the page is written, and
            # the page waits for them only when it is time to put them in.
            failure: list[BaseException] = []

            def draw() -> None:
                try:
                    pictures, seconds[IMAGES] = self._draw(
                        plan.pictures, assignment.images, work_dir, report, on_downloading, on_picture, control
                    )
                    drawn.extend(pictures)
                except BaseException as exc:  # noqa: BLE001 -- re-raised on the conductor's thread below
                    failure.append(exc)
                    report(IMAGES, "failed", assignment.images, str(exc))

            artist = threading.Thread(target=draw, daemon=True, name="page-agent-images")
            artist.start()

            def wait_for_pictures() -> None:
                artist.join()
                if failure:
                    raise failure[0]

            try:
                page, seconds[PAGE] = self._write(
                    prompt, plan, assignment.page, work_dir, report, on_downloading, wait_for_pictures, control
                )
            finally:
                artist.join()
        else:
            # One chip for both: in turn, and never both models in memory.
            if assignment.images == assignment.page:
                self._release_page()
            drawn, seconds[IMAGES] = self._draw(
                plan.pictures, assignment.images, work_dir, report, on_downloading, on_picture, control
            )
            if assignment.images == assignment.page:
                self._release_images()
            page, seconds[PAGE] = self._write(prompt, plan, assignment.page, work_dir, report, on_downloading, None, control)

        report(CHECK, "running", "CPU", "")
        names = [spec.name for spec in plan.pictures]
        written, used = _finished(page, plan)
        verdict = checking.review(written, plan, used, page.html_truncated)
        attempts = 1
        note = checking.repair_note(verdict, self._ran_away)
        if note and not stopped(control):
            report(CHECK, "running", "CPU", "asking for the page once more: " + "; ".join(c.name for c in verdict if not c.passed))
            again, extra = self._write(
                prompt, plan, assignment.page, work_dir, report, on_downloading, None, control, note
            )
            seconds[PAGE] += extra
            rewritten, shown = _finished(again, plan)
            second = checking.review(rewritten, plan, shown, again.html_truncated)
            attempts = 2
            if _passed(second) >= _passed(verdict):
                page, verdict, written, used = again, second, rewritten, shown

        # A picture shown several times, or one nobody drew, is a page asking
        # for more pictures (see repeats.py): they are drawn from the page's
        # own alt texts.
        patched, wanted = give_repeats_their_own(written, names)
        if not wanted:
            written = patched  # nothing to draw; at most an <img> pointing nowhere was taken out
        elif not stopped(control):
            report(CHECK, "running", "CPU", f"the page asks for {len(wanted)} more picture(s)")
            specs = [planning.PictureSpec(name, *planning.BODY_SIZE, _as_a_photograph(alt)) for name, alt in wanted]
            if assignment.images == assignment.page:
                self._release_page()  # in turn, as before: the image model comes back alone
            more, extra = self._draw(
                specs, assignment.images, work_dir, report, on_downloading, on_picture, control, "extra picture"
            )
            seconds[IMAGES] += extra
            if len(more) == len(wanted):  # stopped part-way: the page keeps its repeats rather than broken pictures
                written = patched
                drawn = drawn + more
        every = [
            picture_kit.Picture(picture.name, picture.path, "image/jpeg", picture.width, picture.height, picture.prompt)
            for picture in drawn
        ]
        html, used = picture_kit.embed(written, every)
        verdict = checking.review(written, plan, used, page.html_truncated)
        report(CHECK, "done", "CPU", f"{_passed(verdict)} of {len(verdict)} checks passed")

        seconds["total"] = time.perf_counter() - began
        return PageResult(
            html=html,
            plan=plan,
            assignment=assignment,
            pictures=drawn,
            pictures_used=used,
            checks=verdict,
            attempts=attempts,
            seconds={step: round(value, 1) for step, value in seconds.items()},
            planner_stats=planner_stats,
            stats=page.stats,
            html_source=written,
            cancelled=stopped(control),
        )


def _finished(page, plan: planning.PagePlan) -> tuple[str, list[str]]:
    """The page as the model wrote it, with the studio's layout rules at the
    top of its stylesheet, and the planned pictures it shows."""
    written = art_direction.with_stylesheet(page.html_source or page.html, plan.pictures)
    return written, picture_kit.referenced(written, [spec.name for spec in plan.pictures])


def _passed(verdict: list[Check]) -> int:
    return sum(check.passed for check in verdict)


def _as_a_photograph(description: str) -> str:
    """What the image model is asked for, for a picture the page described
    itself: its alt text, as a photograph. Not in the plan's STYLE, which is
    the page's -- its colours, its type -- and an image model told "an
    elegant serif" draws letters."""
    return f"{description.rstrip('.')}, editorial photograph, soft natural light"
