from torch.optim.lr_scheduler import _LRScheduler


__all__ = [
    "NoamLR",
]


class NoamLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs: int, warmup_factor: float = 1.0):
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        super().__init__(optimizer)

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)
        scale = self.warmup_factor \
              * min(last_epoch ** (-0.5), last_epoch * self.warmup_epochs ** (-1.5))
        return [base_lr * scale for base_lr in self.base_lrs]
