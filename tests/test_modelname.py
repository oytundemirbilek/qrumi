"""Test graph dataset classes."""

from __future__ import annotations

import os


GOLD_STANDARD_PATH = os.path.join(os.path.dirname(__file__), "expected")
DEVICE = "cpu"


def test_simple_iteration() -> None:
    """Test if the model can be iterated - cpu based."""
