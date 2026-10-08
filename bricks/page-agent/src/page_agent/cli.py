"""Command-line entry point for the page agent (experimental)."""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from pantherlake_ai_core import engine as engine_mod
from pantherlake_ai_core import npu

from .samples import SAMPLES


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="page-agent",
        description="EXPERIMENTAL. Build an illustrated, self-contained web page from a request -- one line, or a "
                    "full brief: a small model plans it, an image model draws its pictures, a coding model writes "
                    "the HTML.",
    )
    source = p.add_mutually_exclusive_group(required=False)
    source.add_argument("request", nargs="?", default=None, help="What the page is for: a sentence, or a brief.")
    source.add_argument("--sample", default=None, help="Use a named example request instead (see --list-samples).")
    p.add_argument("--out", default="page.html", help="Where to write the page. Default: page.html")
    p.add_argument("--keep-pictures", default=None, help="Also keep the drawn pictures in this folder.")
    p.add_argument("--planner-device", default=None, help="Chip for the planner. Default: the NPU if there is one.")
    p.add_argument("--image-device", default=None, help="Chip for the image model. Default: the integrated GPU.")
    p.add_argument(
        "--page-device", default=None,
        help="Chip for the coding model. Default: the discrete GPU if there is one, else the integrated GPU "
             "(the two models then take turns on it).",
    )
    p.add_argument("--list-devices", action="store_true", help="List available inference devices, then exit.")
    p.add_argument("--list-samples", action="store_true", help="List available example requests, then exit.")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_devices:
        engine_mod.print_devices()
        return 0
    if args.list_samples:
        for sample in SAMPLES:
            print(f"{sample.name}\n    {sample.description}")
        return 0

    request = args.request
    if args.sample:
        chosen = next((s for s in SAMPLES if s.name.lower().startswith(args.sample.lower())), None)
        if chosen is None:
            parser.error(f"No sample called '{args.sample}' (see --list-samples).")
        request = chosen.prompt
    if not request:
        parser.error("Say what the page is for, or pick one with --sample.")

    from .conductor import PageAgent, assign

    assignment = assign(
        engine_mod.list_openvino_devices(),
        engine_mod.list_gpu_devices(),
        planner=args.planner_device,
        images=args.image_device,
        page=args.page_device,
        npu_usable=not npu.lost(),
    )
    how = "at the same time" if assignment.together else "in turn"
    print(
        f"Plan on {assignment.planner}, pictures on {assignment.images}, page on {assignment.page} "
        f"(pictures and page {how}). The first run may download models.",
        file=sys.stderr,
    )

    def on_step(step: str, state: str, device: str, detail: str) -> None:
        print(f"  [{step}] {state} on {device}{' -- ' + detail if detail else ''}", file=sys.stderr)

    def on_plan(plan) -> None:
        said = " -- ".join(part for part in (plan.title, plan.headline) if part)
        print(f"  [plan] {said or 'no name given'}; on offer: {', '.join(plan.offers) or 'not said'}", file=sys.stderr)

    work_dir = Path(args.keep_pictures) if args.keep_pictures else Path(tempfile.mkdtemp(prefix="page-agent-"))
    result = PageAgent().build(request, assignment=assignment, work_dir=work_dir, on_step=on_step, on_plan=on_plan)

    Path(args.out).write_text(result.html, encoding="utf-8")
    for note in result.plan.notes:
        print(f"  note: {note}", file=sys.stderr)
    for check in result.checks:
        print(f"  {'ok  ' if check.passed else 'FAIL'} {check.name}{': ' + check.detail if check.detail else ''}", file=sys.stderr)
    took = ", ".join(f"{step} {seconds:.0f} s" for step, seconds in result.seconds.items())
    print(f"Wrote {args.out} ({len(result.html) // 1024} KB) -- {took}.", file=sys.stderr)
    return 0 if all(check.passed for check in result.checks) else 2


def run() -> None:
    """The console entry point: `main`, then out without the runtimes' own
    teardown, which after some builds never returns (see leaving.py)."""
    from .leaving import leave_now

    leave_now(main())


if __name__ == "__main__":
    run()
