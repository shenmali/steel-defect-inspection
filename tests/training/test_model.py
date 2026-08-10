import torch

from steel_inspection.training.model import create_model


def test_create_model_returns_four_raw_logit_channels():
    """Detects a model factory that configures fewer than four output classes."""
    model = create_model().eval()

    with torch.no_grad():
        logits = model(torch.zeros((1, 3, 64, 64)))

    assert logits.shape == (1, 4, 64, 64)
