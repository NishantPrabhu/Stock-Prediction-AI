
"""
Evaluation metrics
"""

import torch


def accuracy(output, target):
    assert isinstance(output, torch.Tensor), f"Expected output to be of type torch.Tensor, got {type(output)}"
    assert isinstance(target, torch.Tensor), f"Expected target to be of type torch.Tensor, got {type(target)}"
    preds = output.argmax(dim=-1)
    correct = preds.eq(target.view_as(preds)).sum().item()
    return correct/output.size(0)
