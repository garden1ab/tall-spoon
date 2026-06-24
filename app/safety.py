"""Minimal local prompt guard.

This is intentionally conservative and simple. It is not a substitute for
production moderation. Krea's model license and acceptable-use requirements
remain the deployer's responsibility.
"""

from __future__ import annotations

BLOCKED_TERMS = [
    "csam",
    "child sexual",
    "sexualized child",
    "minor nude",
    "non consensual intimate",
    "non-consensual intimate",
    "revenge porn",
    "deepfake nude",
    "make a nude of",
    "terrorist propaganda",
    "how to make a bomb",
]


def check_prompt(prompt: str) -> tuple[bool, str]:
    p = (prompt or "").lower()
    for term in BLOCKED_TERMS:
        if term in p:
            return False, f"Blocked by local prompt guard: `{term}`"
    return True, "OK"
