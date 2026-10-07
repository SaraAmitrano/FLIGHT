import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


def focal_func(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1,
    gamma: float = 2,
    reduction: str = "mean",
) -> torch.Tensor:
    # Calculate Cross entropy loss
    ce_loss = F.cross_entropy(inputs, targets, reduction="none")

    # Compute pt (model confidence on true class)
    pt = torch.exp(-ce_loss)

    # Apply the focal adjustment
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss

    # Apply reduction (mean, sum, or no reduction)
    if reduction == "mean":
        focal_loss = focal_loss.mean()
    elif reduction == "sum":
        focal_loss = focal_loss.sum()
    return focal_loss


class FocalLoss(nn.Module):
    def __init__(
        self, alpha: float = 1, gamma: float = 2, reduction: str = "mean"
    ) -> None:
        super().__init__()  # type: ignore
        self.alpha = alpha  # controls class imbalance
        self.gamma = gamma  # focuses on hard examples
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return focal_func(
            inputs,
            targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )
