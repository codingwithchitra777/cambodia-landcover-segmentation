"""Losses for the segmentation models.

Two things live here:
  * Plain (optionally class-weighted) cross-entropy — used by the three baselines
    and the standard DeepLabV3+.
  * The **boundary-aware combined loss (Dice + boundary)** — modification 3 of the
    SWIR-Attention DeepLabV3+, used *instead of* plain cross-entropy to sharpen
    the small-parcel edges (see CLAUDE.md).

The boundary term follows the differentiable boundary loss of Bokhovkin &
Burnaev (2019), "Boundary Loss for Remote Sensing Imagery Semantic Segmentation":
it extracts soft class boundaries with pooling ops and optimises a soft boundary
F1, which is directly one of the project's evaluation metrics.

All losses honour `ignore_index` (default 255) so no-data padding never
contributes to the loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = 255


def _one_hot(target, num_classes, ignore_index=IGNORE_INDEX):
    """Return (one_hot (N,C,H,W) float, valid_mask (N,1,H,W) float).

    Ignored pixels are zeroed in the one-hot and marked 0 in the valid mask.
    """
    valid = (target != ignore_index).unsqueeze(1).float()
    safe = target.clone()
    safe[target == ignore_index] = 0
    onehot = F.one_hot(safe, num_classes).permute(0, 3, 1, 2).float()
    return onehot * valid, valid


def dice_loss(logits, target, num_classes, class_weights=None, ignore_index=IGNORE_INDEX, eps=1.0):
    """Soft multiclass Dice loss over valid pixels."""
    probs = F.softmax(logits, dim=1)
    onehot, valid = _one_hot(target, num_classes, ignore_index)
    probs = probs * valid

    dims = (0, 2, 3)
    intersection = (probs * onehot).sum(dims)
    cardinality = probs.sum(dims) + onehot.sum(dims)
    dice = (2 * intersection + eps) / (cardinality + eps)  # per class
    loss = 1.0 - dice

    if class_weights is not None:
        w = class_weights.to(loss.device)
        return (loss * w).sum() / w.sum()
    return loss.mean()


class BoundaryLoss(nn.Module):
    """Soft boundary F1 loss (Bokhovkin & Burnaev, 2019).

    Extracts predicted/target class boundaries via pooling, then computes a soft
    boundary precision/recall and returns 1 - F1.
    """

    def __init__(self, num_classes, theta0=3, theta=5, ignore_index=IGNORE_INDEX):
        super().__init__()
        self.num_classes = num_classes
        self.theta0 = theta0
        self.theta = theta
        self.ignore_index = ignore_index

    @staticmethod
    def _boundary(mask, theta0):
        # Outer boundary of a soft mask via max-pool of its complement.
        b = F.max_pool2d(1 - mask, kernel_size=theta0, stride=1, padding=(theta0 - 1) // 2)
        return b - (1 - mask)

    def forward(self, logits, target):
        probs = F.softmax(logits, dim=1)
        onehot, valid = _one_hot(target, self.num_classes, self.ignore_index)
        probs = probs * valid

        pred_b = self._boundary(probs, self.theta0)
        gt_b = self._boundary(onehot, self.theta0)

        pred_b_ext = F.max_pool2d(pred_b, self.theta, stride=1, padding=(self.theta - 1) // 2)
        gt_b_ext = F.max_pool2d(gt_b, self.theta, stride=1, padding=(self.theta - 1) // 2)

        dims = (0, 2, 3)
        eps = 1e-7
        precision = (pred_b * gt_b_ext).sum(dims) / (pred_b.sum(dims) + eps)
        recall = (gt_b * pred_b_ext).sum(dims) / (gt_b.sum(dims) + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        return (1.0 - f1).mean()


class BoundaryAwareLoss(nn.Module):
    """Combined Dice + boundary loss (modification 3).

    loss = dice_weight * Dice + boundary_weight * Boundary  (+ optional CE for
    early-training stability, off by default to stay a true Dice+boundary loss).
    """

    def __init__(
        self,
        num_classes,
        class_weights=None,
        dice_weight=1.0,
        boundary_weight=1.0,
        ce_weight=0.0,
        ignore_index=IGNORE_INDEX,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.ce_weight = ce_weight
        self.ignore_index = ignore_index
        self.boundary = BoundaryLoss(num_classes, ignore_index=ignore_index)

    def forward(self, logits, target):
        cw = self.class_weights
        loss = self.dice_weight * dice_loss(logits, target, self.num_classes, cw, self.ignore_index)
        loss = loss + self.boundary_weight * self.boundary(logits, target)
        if self.ce_weight > 0:
            loss = loss + self.ce_weight * F.cross_entropy(
                logits, target, weight=cw, ignore_index=self.ignore_index
            )
        return loss


def build_loss(name, num_classes=6, class_weights=None, ignore_index=IGNORE_INDEX):
    """Factory: 'ce' (baselines / standard DeepLabV3+) or 'boundary_aware'
    (SWIR-Attention DeepLabV3+, modification 3)."""
    if name == "ce":
        return nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
    if name == "boundary_aware":
        return BoundaryAwareLoss(num_classes, class_weights=class_weights, ignore_index=ignore_index)
    raise ValueError(f"unknown loss {name!r}; use 'ce' or 'boundary_aware'")


if __name__ == "__main__":
    torch.manual_seed(0)
    N, C, H, W = 2, 6, 64, 64
    logits = torch.randn(N, C, H, W, requires_grad=True)
    target = torch.randint(0, C, (N, H, W))
    target[:, :5, :5] = IGNORE_INDEX  # some no-data padding

    for name in ("ce", "boundary_aware"):
        loss = build_loss(name, num_classes=C)(logits, target)
        loss.backward()
        assert torch.isfinite(loss), f"{name} loss not finite"
        print(f"{name:14s} loss={loss.item():.4f}  grad_finite={torch.isfinite(logits.grad).all().item()}")
        logits.grad = None
    print("losses OK")
