import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture
def tmp_out(tmp_path):
    return str(tmp_path / "OUT")


@pytest.fixture
def tmp_data(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    return str(d)
