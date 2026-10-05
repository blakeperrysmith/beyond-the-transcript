import numpy as np
import pytest
import torch

from btt.export import export_onnx, parity_check
from btt.model import DeliveryNet, example_input


@pytest.mark.parametrize("bidirectional", [True, False])
def test_tract_matches_pytorch(tmp_path, bidirectional):
    torch.manual_seed(0)
    m = DeliveryNet(6, bidirectional=bidirectional).eval()
    m.set_norm(torch.tensor(-8.0), torch.tensor(3.0))
    p = export_onnx(m, tmp_path / "m.onnx")
    xs = np.random.default_rng(0).normal(-8, 3, (4, 1, 64, 300)).astype(np.float32)
    r = parity_check(m, p, xs)
    assert r["passed"], r  # guards the tract GRU issue (#2751): fail loudly if outputs drift


def test_param_count_and_shape():
    m = DeliveryNet(6)
    assert m(example_input()).shape == (1, 6)
    assert sum(p.numel() for p in m.parameters()) == 106_934
