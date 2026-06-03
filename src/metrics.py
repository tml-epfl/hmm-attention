import torch


class LossMetric:
    """Keeps track of the loss over an epoch"""

    def __init__(self) -> None:
        self.running_loss = 0
        self.count = 0

    def update(self, loss: float, batch_size: int = 1) -> None:
        self.running_loss += loss * batch_size
        self.count += batch_size

    def compute(self) -> float:
        if self.count == 0:
            return float("inf")

        return self.running_loss / self.count

    def reset(self) -> None:
        self.running_loss = 0
        self.count = 0


class ConstantLossMetric(LossMetric):
    def reset(self) -> None:
        assert False, "constant metric cannot be reset"


class RelativeMetric(LossMetric):
    """Keeps track of the relative loss over an epoch"""

    def __init__(self, metric: LossMetric) -> None:
        super().__init__()
        self.metric = metric

    def compute(self) -> float:
        return (
            float("inf")
            if self.reference == 0
            else self.metric.compute() / self.reference
        )

    def reset(self) -> None:
        assert False, "RelativeMetric cannot be reset"

    def freeze(self) -> None:
        self.reference = (
            float("inf") if self.count == 0 else self.running_loss / self.count
        )

class AccuracyMetric:
    """Keeps track of the top-k accuracy over an epoch

    Args:
        k (int): Value of k for top-k accuracy
    """

    def __init__(self, k: int = 1) -> None:
        self.correct = 0
        self.total = 0
        self.k = k

    def update(self, out: torch.Tensor, target: torch.Tensor) -> None:
        out = out.reshape(-1, out.shape[-1])
        target = target.reshape(-1, out.shape[-1])

        # Computes top-k accuracy
        target = target.argmax(dim=-1)
        _, indices = torch.topk(out, self.k, dim=-1)
        target_in_top_k = torch.eq(indices, target[:, None]).bool().any(-1)
        total_correct = torch.sum(target_in_top_k, dtype=torch.int).item()
        total_samples = target.shape[0]

        self.correct += total_correct
        self.total += total_samples

    def compute(self) -> float:
        return self.correct / self.total

    def reset(self) -> None:
        self.correct = 0
        self.total = 0


class ConstantAccuracyMetric(AccuracyMetric):
    def reset(self) -> None:
        assert False, "constant metric cannot be reset"


class MinMetric:
    """Keeps track of the minimum value"""

    def __init__(self) -> None:
        self.min = float("inf")

    def update(self, value: float) -> None:
        self.min = min(self.min, value)

    def compute(self) -> float:
        return self.min

    def reset(self) -> None:
        self.min = float("inf")
