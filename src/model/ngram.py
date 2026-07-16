from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.decoder import TransformerDecoder
from src.utils import pad_sequence, split_into_windows


class NgramTransformerDecoder(nn.Module):
    """
    Wrapper around TransformerDecoder that restricts every attention head in
    every layer to see **at most the first *n* tokens** of each sliding-window
    context.

    Setting ``ngram=1`` reproduces the original Unigram behaviour,
    ``ngram=2`` is a bigram model, ``ngram=3`` a trigram model, and so on.
    """

    def __init__(
        self,
        ngram: int,
        **kwargs,
    ):
        super().__init__()
        assert ngram >= 1, "ngram must be ≥ 1"
        self.ngram = ngram
        self.transformer = TransformerDecoder(**kwargs)

    def forward(
        self,
        x: torch.Tensor,  # (batch, seq_len, dim)
        span_lengths: list,
        unroll_sequences: bool = True,
        ngram: Optional[int] = None,
        stride: Optional[int] = None,
    ):
        assert unroll_sequences == True, "`unroll_sequences` = False is unchecked!"

        ngram = ngram or self.ngram

        if ngram > len(span_lengths):
            raise ValueError(
                f"ngram ({ngram}) cannot be greater than number of span_lengths ({len(span_lengths)})"
            )

        # compute the window size of the model
        if stride is not None:
            # With stride: (ngram - 1) * stride + last_span_length
            window_size = (ngram - 1) * stride + span_lengths[ngram - 1]
        else:
            # Without stride: sum of first ngram spans
            window_size = sum(span_lengths[:ngram])

        if unroll_sequences:
            bsz, _, dim = x.shape
            tokens, targets = split_into_windows(x, window_size, pad=0)
        else:
            tokens = pad_sequence(x, pad=0)

        logits, _ = self.transformer(tokens)  # (batch * num_windows, total_window, dim)

        logits = logits[:, -1, :]

        # Unroll back to original sequence shape if required
        if unroll_sequences:
            logits = logits.view(bsz, -1, dim)
            targets = targets.view(bsz, -1, dim)

        probabilities = F.softmax(logits, dim=-1)
        return logits, probabilities, targets if unroll_sequences else None
