from abc import ABC, abstractmethod
import math

import torch, torch.nn.functional as F
from torch.utils.data import Dataset


from src.model import TransformerDecoder
from src.teachers import ARTeacher
from src.utils import split_into_windows


class ARRegression(Dataset, ABC):
    def __init__(
        self,
        teacher: torch.nn.Module,
        window: int,
        dim: int,
        number: int,
        length: int,
        prefix_length: int,
        unroll_sequences: bool = True,
        random_sampling: bool = False,
        replicate_context_for_spans: bool = True,
    ) -> None:
        if prefix_length == -1:
            raise ValueError(
                "prefix_length must be provided as sum(teacher.span_lengths)."
            )

        """Creates an autoregressive model with `number` sequences of length `length`
        with context lenght w of d dimensional tokens."""
        self.teacher = teacher
        self.window = window
        self.dim = dim
        self.number = abs(number)
        self.length = length
        self.unroll_sequences = unroll_sequences
        self.refresh_sequences = number < 0
        self.random_sampling = random_sampling
        self.replicate_context_for_spans = replicate_context_for_spans
        self.prefix_length = prefix_length
        self.data = self._generate(self.number, self.length)

    @abstractmethod
    def _generate_prefix(self, length: int) -> torch.Tensor:
        """Generate an initial prefix of length w."""
        raise NotImplementedError

    @abstractmethod
    def _error_model(self, prefix: torch.Tensor) -> torch.Tensor:
        """Returns the error for the given prefix."""
        raise NotImplementedError

    def _sampling_model(self, embedding: torch.Tensor) -> torch.Tensor:
        """Returns sampled output token for the given embedding."""
        return embedding

    def _autoregressive_model(self, prefix: torch.Tensor) -> torch.Tensor:
        """
        Returns the teacher's prediction for the *next* token given `prefix`.
        - ARTeacher.predict_next auto-slices to context_length and returns (B, D).
        - TransformerDecoder returns (B, L, D) over the whole prefix; we take last step.
        """
        with torch.no_grad():
            device = next(self.teacher.parameters()).device

            if isinstance(self.teacher, ARTeacher):
                y = self.teacher.predict_next(prefix.to(device))
                return y.to(prefix.device)  # (B, D)

            if isinstance(self.teacher, TransformerDecoder):
                x = prefix.unsqueeze(0).to(device)  # (1, K, D)
                out = self.teacher(x)
                logits = out[0] if isinstance(out, tuple) else out  # (1, K, D)
                next_vec = logits[:, -1, :]  # (1, D) — prediction for next token
                return next_vec.squeeze(0).to(prefix.device)  # (D,)

            raise NotImplementedError(
                f"Teacher model of type {type(self.teacher)} not supported."
            )

    def _generate_seq(self, length: int) -> torch.Tensor:
        """Generates a sequence of length `length` with a prefix."""
        # Create empty sequence with space for initial prefix tokens and data tokens (length).
        seq = torch.zeros((self.prefix_length + length, self.dim))

        # Generate prefix tokens
        if self.replicate_context_for_spans:
            # Strategy 1: Replicate context tokens across spans
            prefix_tokens = self._generate_prefix(self.window)
            for i in range(self.prefix_length):
                # Use modulo to repeat the context pattern across spans.
                context_idx = i % self.window
                seq[i, :] = self._sampling_model(prefix_tokens[context_idx, :])
        else:
            # Strategy 2: Sample all prefix tokens independently at random.
            prefix_tokens = self._generate_prefix(self.prefix_length)
            seq[: self.prefix_length, :] = prefix_tokens
            for i in range(self.prefix_length):
                seq[i, :] = self._sampling_model(seq[i, :])

        # Generate the rest of the sequence seeded by the prefix.
        # Always pass the full growing prefix; each teacher handles its own
        # context-slicing (ARTeacher.predict_next auto-slices to context_length,
        # TransformerDecoder uses causal attention over the whole input).
        for i in range(self.prefix_length, seq.shape[0]):
            prefix = seq[:i, :]
            seq[i, :] = self._sampling_model(
                self._autoregressive_model(prefix.unsqueeze(0))  # batch dim = 1
                + self._error_model(prefix)
            )
        return seq

    def _generate(self, number: int, length: int) -> torch.Tensor:
        """Generates `number` sequences of length `length`."""
        return torch.stack([self._generate_seq(length) for _ in range(number)])

    def __len__(self):
        return self.number

    def __getitem__(self, index: int) -> torch.Tensor:
        """Returns the index-th sequence."""
        if self.refresh_sequences:
            self.data[index : index + 1, ...] = self._generate(1, self.length)

        if self.unroll_sequences:
            return split_into_windows(
                torch.unsqueeze(self.data[index, ...], dim=0), self.window
            )
        else:
            return self.data[index, ...]

class ARClassification(ARRegression, ABC):
    def __init__(
        self,
        teacher: torch.nn.Module,
        window: int,
        dim: int,
        number: int,
        length: int,
        softmax: bool = True,
        temperature: float = 1,
        one_hot: bool = False,
        unroll_sequences: bool = True,
        random_sampling: bool = False,
        replicate_context_for_spans: bool = True,
        prefix_length: int = None,
    ) -> None:
        self.softmax = softmax
        self.temperature = temperature
        self.one_hot = one_hot
        self.random_sampling = random_sampling

        super().__init__(
            teacher=teacher,
            window=window,
            dim=dim,
            number=number,
            length=length,
            prefix_length=prefix_length,
            unroll_sequences=unroll_sequences,
            random_sampling=random_sampling,
            replicate_context_for_spans=replicate_context_for_spans,
        )

        assert not (self.refresh_sequences and not self.one_hot), (
            "Use `one_hot` with refreshed sequences (number < 0)!"
        )

    def _sampling_model(self, embedding: torch.Tensor) -> torch.Tensor:
        """Returns a sampled output token for the given embedding."""
        if self.random_sampling:
            # Randomly sample an index.
            sampled_indices = torch.randint(0, self.dim, (1,), device=embedding.device)
        else:
            if self.temperature == -1:
                # Argmax sampling (greedy).
                sampled_indices = torch.tensor(
                    [torch.argmax(embedding, dim=-1)], device=embedding.device
                )
            else:
                # Multinomial sampling based on probabilities.
                probabilities = (
                    F.softmax(embedding / self.temperature, dim=-1)
                    if self.softmax
                    else embedding / torch.linalg.norm(embedding, ord=1, dim=-1)
                )
                sampled_indices = torch.multinomial(
                    probabilities,
                    num_samples=1,
                    replacement=True,
                )

        # Convert the sampled index to a one-hot encoded tensor.
        return F.one_hot(sampled_indices.squeeze(0), num_classes=self.dim).type(
            torch.float
        )


class GaussianARRegression(ARRegression):
    def __init__(
        self,
        teacher: torch.nn.Module,
        window: int,
        dim: int,
        number: int,
        length: int,
        sigma_err: float = 1.0,
        sigma_data: float = 0.0,
        unroll_sequences: bool = True,
        random_sampling: bool = False,
        replicate_context_for_spans: bool = True,
        prefix_length: int = None,
    ) -> None:
        """Creates a autoregressive model where tokens are Gaussian."""
        self.sigma_err = sigma_err
        self.sigma_data = sigma_data
        super().__init__(
            teacher=teacher,
            window=window,
            dim=dim,
            number=number,
            length=length,
            prefix_length=prefix_length,
            unroll_sequences=unroll_sequences,
            random_sampling=random_sampling,
            replicate_context_for_spans=replicate_context_for_spans,
        )

    def _generate_prefix(self, length: int) -> torch.Tensor:
        """Generates an initial prefix of length w."""
        samples = torch.stack(
            [
                torch.randn(self.dim) * (self.sigma_data / math.sqrt(self.dim))
                for _ in range(length)
            ]
        )
        return samples

    def _error_model(self, prefix: torch.Tensor) -> torch.Tensor:
        """Returns the error for the given prefix."""
        return torch.randn(self.dim) * (self.sigma_err / math.sqrt(self.dim))


class GaussianARClassification(ARClassification, GaussianARRegression):
    def __init__(
        self,
        teacher: torch.nn.Module,
        window: int,
        dim: int,
        number: int,
        length: int,
        sigma_err: float = 0.0,
        sigma_data: float = 0.0,
        softmax: bool = True,
        temperature: float = 1,
        one_hot: bool = False,
        unroll_sequences: bool = True,
        random_sampling: bool = False,
        replicate_context_for_spans: bool = True,
        prefix_length: int = None,
    ) -> None:
        """Creates a linear autoregressive model with n sequences
        of length `length` with context lenght w of d dimensional tokens."""
        self.softmax = softmax
        self.temperature = temperature
        self.one_hot = one_hot
        self.random_sampling = random_sampling

        GaussianARRegression.__init__(
            self,
            teacher=teacher,
            window=window,
            dim=dim,
            number=number,
            length=length,
            prefix_length=prefix_length,
            sigma_err=sigma_err,
            sigma_data=sigma_data,
            unroll_sequences=unroll_sequences,
            random_sampling=random_sampling,
            replicate_context_for_spans=replicate_context_for_spans,
        )

