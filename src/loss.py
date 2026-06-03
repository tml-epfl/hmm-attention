import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    __constants__ = ["reduction"]

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.square(input - target))


class CrossentropyLoss(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input = input.reshape(-1, input.shape[-1])
        target = target.reshape(-1, target.shape[-1])
        return self.loss_fn(input, target)


class KLDivergenceLoss(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the KL divergence loss.

        Args:
            input (Tensor): The logits from the transformer of shape (batch_size, seq_length, dim).
                            These are raw scores (not probabilities).
            target (Tensor): The target probability distributions of shape (batch_size, seq_length, dim).
                             Each slice over the last dimension should sum to 1.

        Returns:
            Tensor: The computed KL divergence loss.

        The loss for each position is computed as:
            KL(p || q) = sum_i p(i) * (log p(i) - log q(i))
        where log q is obtained by applying log_softmax on the logits (input).
        """
        # Clamp target to avoid log(0)
        target = target.clamp(min=1e-10)

        # Convert logits to log-probabilities using log_softmax on the last dimension
        log_q = F.log_softmax(input, dim=-1)

        # Calculate KL divergence per example per time step: p * (log(p) - log(q))
        loss = target * (torch.log(target) - log_q)

        # Sum over the class dimension (dim), resulting in a tensor of shape (batch_size, seq_length)
        loss = loss.sum(dim=-1)

        # Apply reduction over batch and sequence dimensions if needed
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # "none"
            return loss
