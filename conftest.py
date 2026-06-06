"""Pytest configuration.

Marks
-----
slow    Tests that load the full MERT model (requires a network download on
        first run and ~3 GB RAM).  Skipped by default.

Run slow tests explicitly::

    pytest -m slow
"""
import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: marks tests that load the full MERT model (skip with -m 'not slow')",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip @pytest.mark.slow tests unless -m slow is explicitly passed."""
    if config.getoption("-m", default="") == "slow":
        return  # user opted in — run them

    skip_slow = pytest.mark.skip(reason="Use -m slow to run MERT integration tests")
    for item in items:
        if item.get_closest_marker("slow"):
            item.add_marker(skip_slow)
