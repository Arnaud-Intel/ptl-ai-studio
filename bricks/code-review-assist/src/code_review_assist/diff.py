"""Reads a diff to review, either by running `git diff` in a working tree
or from diff text supplied directly (e.g. pasted into the launcher UI).

No brick in this workspace shells out to git today -- this is the first,
so it's deliberately conservative: a timeout on the subprocess call, no
shell interpolation (argv list, not a shell string), and flags that stop a
user's gitconfig (color, external diff tools) from corrupting the captured
text.
"""
from __future__ import annotations

import subprocess

MAX_DIFF_CHARS = 12_000


def read_diff(*, folder: str | None = None, against: str = "HEAD", diff_text: str | None = None) -> str:
    if diff_text is not None:
        return diff_text.strip()

    if folder:
        try:
            result = subprocess.run(
                ["git", "-C", folder, "diff", "--no-color", "--no-ext-diff", against],
                capture_output=True,
                text=True,
                timeout=20,
            )
        except FileNotFoundError:
            raise RuntimeError("git isn't installed or isn't on PATH.") from None
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"`git diff` in '{folder}' took too long (over 20s).") from None
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"`git diff` failed in '{folder}'.")
        return result.stdout.strip()

    raise ValueError("Provide either a folder (to run `git diff` in) or diff text.")


def truncate_diff(text: str, max_chars: int = MAX_DIFF_CHARS) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True
