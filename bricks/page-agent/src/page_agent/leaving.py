"""Ending a process that has built a page.

After some builds the process does not exit. Seen on the XPS 14 on
2026-10-08, five times out of five, with one particular build: the bike
sample, with pictures and page sharing the integrated GPU, where the image
model is loaded, released, the coding model loaded, released, and the image
model loaded again for four extra pictures. Everything is released in under
a second; Python tears down every module -- its verbose log ends on "clear
sys.audit hooks", the last thing it does -- and then one thread spins at
100% in the runtimes' own teardown, fourteen minutes and counting when it
was killed. Eight shorter sequences of the same loads and releases, some
with the same models and the same long page, all exited at once: what sets
it off was not found.

It is past anything this code can reach, so a process that has built a page
ends itself here, once everything of its own is done and flushed, without
that teardown.
"""
from __future__ import annotations

import os
import sys


def leave_now(code: int = 0) -> None:
    """Flush what was printed and end the process with `code`, skipping the
    interpreter's and the runtimes' teardown. For the very end only: nothing
    after this runs, `finally` blocks and exit handlers included.

    On Windows `os._exit` is not enough: it ends in ExitProcess, which still
    tells every loaded DLL the process is going -- and it is in one of
    those farewells that the thread spins (tried: the process hung exactly
    as before). TerminateProcess ends it without asking anyone."""
    sys.stdout.flush()
    sys.stderr.flush()
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        kernel32.TerminateProcess(kernel32.GetCurrentProcess(), code)
    os._exit(code)
