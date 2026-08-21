"""Shared building blocks for the Panther Lake local AI demo bricks.

Every brick in this monorepo is a standalone, installable package, but they
share a small common core so that things like "capture audio", "detect
speech", and "which local compute backend/device should this use" aren't
reimplemented per demo.
"""

__version__ = "0.1.0"
