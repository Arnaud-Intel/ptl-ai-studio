"""Named example prompts so the CLI/launcher can offer a ready-to-run demo
landing page instead of requiring the user to write their own prompt."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Sample:
    name: str
    description: str
    prompt: str


SAMPLES: list[Sample] = [
    Sample(
        name="Coffee shop landing page",
        description="A cozy local-business page with a menu highlight.",
        prompt=(
            "A warm, cozy landing page for 'Maple & Rye', a neighborhood coffee shop. "
            "Include a hero section with the tagline 'Small batch. Big mornings.', a "
            "short story about the roastery, a menu highlight section (pour-over, cold "
            "brew, seasonal lattes), and a footer with the address and hours."
        ),
    ),
    Sample(
        name="SaaS product page",
        description="A clean product page with pricing tiers.",
        prompt=(
            "A clean, modern landing page for 'Lumen', a project-tracking SaaS for "
            "small teams. Hero section with headline 'Ship on time, every time', three "
            "feature cards (kanban boards, automated status reports, Slack "
            "integration), a pricing section with three tiers (Free, Team, Business), "
            "and a call-to-action button that says 'Start free trial'."
        ),
    ),
    Sample(
        name="Personal portfolio",
        description="A minimal one-page photographer portfolio.",
        prompt=(
            "A minimal one-page portfolio for a freelance photographer named Alex "
            "Rivera. Include a full-width hero image placeholder, an 'About' section, "
            "a grid gallery placeholder with 6 photo slots, and a contact section with "
            "an email link."
        ),
    ),
    Sample(
        name="Nonprofit fundraiser",
        description="A donation page with a progress bar toward a goal.",
        prompt=(
            "A landing page for a local nonprofit's fundraiser 'Books for Every "
            "Block', which collects used children's books for underfunded schools. "
            "Include a donation call-to-action, a progress bar toward a 5,000-book "
            "goal, and a short section explaining how to donate books in person."
        ),
    ),
]
