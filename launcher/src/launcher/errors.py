"""The one exception a runner raises for a *state* problem -- "already
running", "enroll a voice first", "stop recording before resetting" -- as
opposed to a bad input (ValueError/FileNotFoundError) or a failure inside
a model (anything else). app.py maps the three to 409 / 400 / 500.

A RuntimeError subclass, so any caller that already catches RuntimeError
keeps working.
"""
from __future__ import annotations


class Conflict(RuntimeError):
    """The brick isn't in a state where this request makes sense."""
