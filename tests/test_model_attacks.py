"""Dual BN, attacks' constraints, DAT generator shape (CLAUDE.md §11)."""
import torch

from src import attacks
from src.aag import AmplitudeGenerator
from src.models import ResNet18


def _model(dual=True):
    return ResNet18(num_classes=43, dual_bn=dual).eval()


def test_dual_bn_branches_differ_and_test_uses_A():
    m = _model(True)
    # perturb BN-B parameters so the two branches differ
    with torch.no_grad():
        for mod in m._bns:
            mod.bn_b.weight.mul_(2.0)
            mod.bn_b.bias.add_(0.5)
            mod.bn_b.running_mean.add_(1.0)
    x = torch.rand(4, 3, 32, 32)
    la = m(x, bn="A")
    lb = m(x, bn="B")
    assert not torch.allclose(la, lb), "branches A and B produce identical outputs"
    # a plain call (as AutoAttack / evaluation do) equals BN-A
    assert torch.allclose(m(x), la)
    # after a B forward the model is left routed to A
    m(x, bn="B")
    assert all(mod.route == "A" for mod in m._bns)
    # parameters are distinct tensors
    p_a = {id(p) for mod in m._bns for p in mod.bn_a.parameters()}
    p_b = {id(p) for mod in m._bns for p in mod.bn_b.parameters()}
    assert not (p_a & p_b)


def test_single_bn_ignores_route():
    m = _model(False)
    x = torch.rand(2, 3, 32, 32)
    assert torch.allclose(m(x, bn="A"), m(x, bn="B"))


def test_feature_map_and_cam_shapes():
    m = _model()
    x = torch.rand(3, 3, 32, 32)
    logits, feat = m(x, return_feat=True)
    assert logits.shape == (3, 43) and feat.shape == (3, 512, 4, 4)
    cam = m.cam(feat, torch.tensor([0, 5, 42]))
    assert cam.shape == (3, 16)


def _check_ball(x_adv, x, eps):
    assert (x_adv - x).abs().max() <= eps + 1e-6
    assert x_adv.min() >= 0 and x_adv.max() <= 1


def test_pgd_eaeg_fgsm_cw_within_ball():
    m = _model(False)
    x = torch.rand(8, 3, 32, 32)
    y = torch.randint(0, 43, (8,))
    eps, alpha = 8 / 255, 2 / 255
    _check_ball(attacks.pgd(m, x, y, eps, alpha, 5), x, eps)
    _check_ball(attacks.pgd(m, x, y, eps, alpha, 5, loss="cw"), x, eps)
    _check_ball(attacks.fgsm(m, x, y, eps), x, eps)
    _check_ball(attacks.eaeg(m, x, y, eps, alpha, steps=5, beta=15.0), x, eps)
    _check_ball(attacks.trades_inner(m, x, eps, alpha, steps=5), x, eps)
    m.train()
    attacks.eaeg(m, x, y, eps, alpha, steps=2)
    assert m.training, "attack must restore train mode"


def test_patch_attack_only_changes_patch():
    m = _model(False)
    x = torch.rand(4, 3, 32, 32)
    y = torch.randint(0, 43, (4,))
    idx = [0, 1, 2, 3]
    side = 6
    x_adv = attacks.patch_attack(m, x, y, idx, side, steps=3)
    changed = (x_adv != x).any(1)
    for b, (r, c) in enumerate(attacks.patch_locations(idx, side)):
        assert 4 <= r <= 28 - side and 4 <= c <= 28 - side
        outside = changed[b].clone()
        outside[r:r + side, c:c + side] = False
        assert not outside.any()
    assert x_adv.min() >= 0 and x_adv.max() <= 1


def test_generator_output_shape_and_range():
    G = AmplitudeGenerator(num_classes=43)
    z = torch.randn(5, 100)
    logits = torch.randn(5, 43)
    out = G(z, logits)
    assert out.shape == (5, 3, 32, 32)
    assert out.min() >= 0 and out.max() <= 1
    linears = [m for m in G.model if isinstance(m, torch.nn.Linear)]
    assert [l.out_features for l in linears] == [128, 256, 512, 1024, 3 * 32 * 32]
