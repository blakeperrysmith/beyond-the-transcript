import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session")
def synthetic_artifacts(tmp_path_factory):
    """Train a few epochs on synthetic audio. Exercises the whole pipeline, proves nothing about speech."""
    from btt.train import main

    root = tmp_path_factory.mktemp("syn")
    out = root / "art"
    main(["--synthetic", str(root / "corpus"), "--out", str(out), "--epochs", "3", "--mode", "main"])
    return out
