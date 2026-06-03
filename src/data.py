from abc import ABC, abstractmethod
import math

import torch, torch.nn.functional as F
from torch.utils.data import Dataset


from src.model import LinearARModel, TransformerDecoder
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
        use_full_prefix: bool = False,
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
        self.use_full_prefix = use_full_prefix
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
        - Linear AR teachers return a single (D,) vector directly.
        - TransformerDecoder returns (B, L, D) over the whole prefix; we take the last step.
        """
        with torch.no_grad():
            device = next(self.teacher.parameters()).device

            # Linear AR Teacher
            if isinstance(self.teacher, LinearARModel):
                y = self.teacher.forward(prefix.to(device), unroll_sequences=False)
                return y.to(prefix.device)  # shape (D,)

            # Transformer Teacher: feed the prefix as a batch of size 1, take last step
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
        for i in range(self.prefix_length, seq.shape[0]):
            if self.use_full_prefix:
                prefix = seq[:i, :]
            else:
                prefix = seq[max(0, i - self.prefix_length) : i, :]
            # Generate next token using teacher prediction + error (error is set to zero.)
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
        use_full_prefix: bool = False,
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
            use_full_prefix=use_full_prefix,
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


class HierarchicalARClassification(ARClassification):
    def __init__(
        self,
        teacher: torch.nn.Module,
        window: int,
        dim: int,
        chunk_dim: int,
        chunk_size: int,
        number: int,
        length: int,
        prefix_length: int,
        softmax: bool = True,
        temperature: float = 1.0,
        one_hot: bool = True,
        unroll_sequences: bool = True,
        random_sampling: bool = False,
        use_full_prefix: bool = False,
        replicate_context_for_spans: bool = True,
        stochastic_emission: bool = False,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if chunk_size > chunk_dim:
            raise ValueError(
                "chunk_size cannot exceed chunk_dim when emitting unique one-hot chunks."
            )
        if not one_hot:
            raise ValueError("Hierarchical autoregressive data requires one_hot=True.")

        super().__init__(
            teacher=teacher,
            window=window,
            dim=dim,
            number=number,
            length=length,
            prefix_length=prefix_length,
            softmax=softmax,
            temperature=temperature,
            one_hot=one_hot,
            unroll_sequences=unroll_sequences,
            random_sampling=random_sampling,
            use_full_prefix=use_full_prefix,
            replicate_context_for_spans=replicate_context_for_spans,
        )

        self.chunk_dim = chunk_dim
        self.chunk_size = chunk_size
        self.stochastic_emission = stochastic_emission
        self._chunks = self._generate_unique_chunks()

    def __getitem__(self, index: int) -> torch.Tensor:
        item = super().__getitem__(index)

        if isinstance(item, tuple):
            xs, ys = item
            return self._replace_with_chunks(xs), self._replace_with_chunks(ys)

        return self._replace_with_chunks(item)

    def _replace_with_chunks(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 0:
            raise ValueError("Input tensor must have at least one dimension.")

        original_device = tensor.device
        flattened = tensor.reshape(-1, tensor.shape[-1])
        if not torch.allclose(
            flattened.sum(dim=-1), torch.ones(flattened.size(0), device=flattened.device)
        ):
            raise ValueError("Hierarchical replacement expects one-hot encoded tokens.")

        token_ids = torch.argmax(flattened, dim=-1)

        vocab_per_h = self._chunks.to(original_device)[token_ids]
        if self.stochastic_emission:
            slots = torch.randint(
                0, self.chunk_size, vocab_per_h.shape[:2], device=original_device
            )
            chunks = vocab_per_h.gather(
                1, slots.unsqueeze(-1).expand(-1, -1, self.chunk_dim)
            )
        else:
            chunks = vocab_per_h
        prefix_shape = tensor.shape[:-1]
        chunks = chunks.view(*prefix_shape, self.chunk_size, self.chunk_dim)

        if prefix_shape:
            new_seq_len = prefix_shape[-1] * self.chunk_size
            return chunks.reshape(*prefix_shape[:-1], new_seq_len, self.chunk_dim)

        return chunks.reshape(self.chunk_size, self.chunk_dim)

    def _generate_unique_chunks(self) -> torch.Tensor:
        """Map each base one-hot vector to a unique sequence of one-hot chunks."""
        total_permutations = math.perm(self.chunk_dim, self.chunk_size)
        if total_permutations < self.dim:
            raise ValueError(
                "Not enough unique chunk permutations to cover all base tokens. "
                f"Need {self.dim}, but only {total_permutations} available."
            )

        chunks = torch.zeros(self.dim, self.chunk_size, self.chunk_dim)
        used_sequences = set()
        for token_idx in range(self.dim):
            attempts = 0
            while True:
                indices = torch.randperm(self.chunk_dim)[: self.chunk_size]
                sequence_signature = tuple(indices.tolist())
                if sequence_signature not in used_sequences:
                    used_sequences.add(sequence_signature)
                    chunk = torch.zeros(self.chunk_size, self.chunk_dim)
                    chunk[torch.arange(self.chunk_size), indices] = 1.0
                    chunks[token_idx] = chunk
                    break

                attempts += 1
                if attempts > 1000:
                    raise RuntimeError(
                        "Failed to sample a unique chunk sequence after many attempts."
                    )

        return chunks

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
        use_full_prefix: bool = False,
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
            use_full_prefix=use_full_prefix,
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
        use_full_prefix: bool = False,
        replicate_context_for_spans: bool = True,
        prefix_length: int = None,
    ) -> None:
        """Creates a linear autoregressive model with n sequences
        of length `length` with context lenght w of d dimensional tokens."""
        self.softmax = softmax
        self.temperature = temperature
        self.one_hot = one_hot
        self.random_sampling = random_sampling
        self.use_full_prefix = use_full_prefix

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
            use_full_prefix=use_full_prefix,
            replicate_context_for_spans=replicate_context_for_spans,
        )

class HierarchicalGaussianARClassification(
    HierarchicalARClassification, GaussianARRegression
):
    def __init__(
        self,
        teacher: torch.nn.Module,
        window: int,
        dim: int,
        chunk_dim: int,
        chunk_size: int,
        number: int,
        length: int,
        prefix_length: int,
        sigma_err: float = 0.0,
        sigma_data: float = 0.0,
        softmax: bool = True,
        temperature: float = 1.0,
        one_hot: bool = True,
        unroll_sequences: bool = True,
        random_sampling: bool = False,
        use_full_prefix: bool = False,
        replicate_context_for_spans: bool = True,
        stochastic_emission: bool = False,
    ) -> None:
        """Hierarchical autoregressive dataset with Gaussian noise on teacher outputs."""
        if not one_hot:
            raise ValueError("Hierarchical Gaussian classification requires one_hot=True.")

        self.sigma_err = sigma_err
        self.sigma_data = sigma_data

        super().__init__(
            teacher=teacher,
            window=window,
            dim=dim,
            chunk_dim=chunk_dim,
            chunk_size=chunk_size,
            number=number,
            length=length,
            prefix_length=prefix_length,
            softmax=softmax,
            temperature=temperature,
            one_hot=one_hot,
            unroll_sequences=unroll_sequences,
            random_sampling=random_sampling,
            use_full_prefix=use_full_prefix,
            replicate_context_for_spans=replicate_context_for_spans,
            stochastic_emission=stochastic_emission,
        )