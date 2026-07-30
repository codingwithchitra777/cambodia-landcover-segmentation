"""Segmentation metrics — confusion-matrix based IoU.

mIoU is the project's headline metric (never pixel accuracy — see CLAUDE.md).
Accumulate predictions over a whole split with `ConfusionMatrix`, then read off
per-class IoU and mean IoU. Boundary F1 lives in `evaluate.py`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

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


def _class_boundary(mask):
    """1-pixel boundary band of a binary mask (N,1,H,W) via max/min pooling."""
    mp = F.max_pool2d(mask, 3, stride=1, padding=1)
    mn = -F.max_pool2d(-mask, 3, stride=1, padding=1)
    return (mp != mn).float()


class BoundaryF1:
    """Streaming boundary F1 with a pixel tolerance.

    A predicted boundary pixel counts as correct if it lies within `tolerance`
    pixels of a ground-truth boundary (and vice-versa for recall). Precision and
    recall are accumulated per class over the whole split, then combined — a
    boundary-quality metric that complements region-based mIoU (see CLAUDE.md).
    """

    def __init__(self, num_classes, ignore_index=IGNORE_INDEX, tolerance=2):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.tolerance = tolerance
        self.tp_p = torch.zeros(num_classes)   # pred-boundary pixels near a GT boundary
        self.sum_p = torch.zeros(num_classes)  # total pred-boundary pixels
        self.tp_r = torch.zeros(num_classes)   # GT-boundary pixels near a pred boundary
        self.sum_r = torch.zeros(num_classes)  # total GT-boundary pixels

    @torch.no_grad()
    def update(self, pred, target):
        """pred, target: (N,H,W) integer label maps."""
        if pred.dim() == 2:
            pred, target = pred.unsqueeze(0), target.unsqueeze(0)
        pred, target = pred.cpu(), target.cpu()
        valid = (target != self.ignore_index).unsqueeze(1).float()
        k = 2 * self.tolerance + 1
        for c in range(self.num_classes):
            pm = (pred == c).unsqueeze(1).float() * valid
            gm = (target == c).unsqueeze(1).float() * valid
            pb, gb = _class_boundary(pm), _class_boundary(gm)
            pb_dil = F.max_pool2d(pb, k, stride=1, padding=self.tolerance)
            gb_dil = F.max_pool2d(gb, k, stride=1, padding=self.tolerance)
            self.tp_p[c] += (pb * gb_dil).sum()
            self.sum_p[c] += pb.sum()
            self.tp_r[c] += (gb * pb_dil).sum()
            self.sum_r[c] += gb.sum()

    def per_class_f1(self):
        eps = 1e-7
        precision = self.tp_p / (self.sum_p + eps)
        recall = self.tp_r / (self.sum_r + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        present = self.sum_r > 0  # class has boundaries in the ground truth
        return f1, present

    def mean_f1(self):
        f1, present = self.per_class_f1()
        if present.sum() == 0:
            return 0.0
        return f1[present].mean().item()
