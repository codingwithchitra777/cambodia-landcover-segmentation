"""Segmentation metrics — confusion-matrix based IoU.

mIoU is the project's headline metric (never pixel accuracy — see CLAUDE.md).
Accumulate predictions over a whole split with `ConfusionMatrix`, then read off
per-class IoU and mean IoU. Boundary F1 lives in `evaluate.py`.
"""

from __future__ import annotations

import torch

IGNORE_INDEX = 255


class ConfusionMatrix:
    """Streaming confusion matrix for semantic segmentation."""

    def __init__(self, num_classes, ignore_index=IGNORE_INDEX):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.mat = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    @torch.no_grad()
    def update(self, pred, target):
        """pred, target: integer label maps of the same shape (any dims)."""
        pred = pred.flatten().cpu()
        target = target.flatten().cpu()
        keep = target != self.ignore_index
        pred, target = pred[keep], target[keep]
        idx = target * self.num_classes + pred
        binc = torch.bincount(idx, minlength=self.num_classes ** 2)
        self.mat += binc.reshape(self.num_classes, self.num_classes)

    def per_class_iou(self):
        """Return (iou tensor [C], valid mask [C]) — valid where the class appears."""
        m = self.mat.float()
        inter = torch.diag(m)
        union = m.sum(1) + m.sum(0) - inter
        iou = inter / union.clamp(min=1e-9)
        return iou, union > 0

    def mean_iou(self):
        """mIoU over classes that appear in the ground truth."""
        iou, valid = self.per_class_iou()
        if valid.sum() == 0:
            return 0.0
        return iou[valid].mean().item()

    def pixel_accuracy(self):
        """Overall accuracy — diagnostic only, never a headline metric."""
        m = self.mat.float()
        return (torch.diag(m).sum() / m.sum().clamp(min=1e-9)).item()
