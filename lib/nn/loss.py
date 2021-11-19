import torch
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss


__all__ = [
    "CrossEntropyLoss",
    "SmoothCrossEntropyLoss",
]


class CrossEntropyLoss(_Loss):
    def __init__(
        self,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__(reduction=reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, input, target):
        is_ignore_index_enabled = (0 <= self.ignore_index < input.size(1))
        num_wrong_labels = input.size(1) - (2 if is_ignore_index_enabled else 1)
        smoothed_target = torch.full_like(input, self.label_smoothing / num_wrong_labels)
        smoothed_target.scatter_(1, target.unsqueeze(1), 1 - self.label_smoothing)
        if is_ignore_index_enabled:
            smoothed_target[:, self.ignore_index] = 0
        out = -torch.sum(F.log_softmax(input, dim=1) * smoothed_target, dim=1)
        if self.reduction == "mean":
            out = torch.mean(out)
        elif self.reduction == "sum":
            out = torch.sum(out)
        return out


# PyTorch 1.10.0 supports label smoothing in `CrossEntropyLoss`
class SmoothCrossEntropyLoss(CrossEntropyLoss):
    pass
