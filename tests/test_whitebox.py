import pytest

torch = pytest.importorskip("torch")

from aiblink_validation.whitebox import pgd_linf


def test_pgd_increases_per_sample_binary_loss():
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 1, bias=False))
    with torch.no_grad():
        model[1].weight.fill_(0.1)
    clean = torch.full((2, 3, 4, 4), 0.5)
    labels = torch.tensor([0.0, 1.0])
    mean = torch.zeros(1, 3, 1, 1)
    std = torch.ones(1, 3, 1, 1)
    with torch.no_grad():
        before = torch.nn.functional.binary_cross_entropy_with_logits(
            model(clean).squeeze(-1), labels, reduction="none"
        )
    attacked = pgd_linf(model, clean, labels, 2 / 255, 4, 1, mean, std)
    with torch.no_grad():
        after = torch.nn.functional.binary_cross_entropy_with_logits(
            model(attacked).squeeze(-1), labels, reduction="none"
        )
    assert torch.all(after >= before)
    assert torch.max(torch.abs(attacked - clean)) <= 2 / 255 + 1e-7
