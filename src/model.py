import math
from abc import ABC, abstractmethod
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from src.utils import (
    pad_sequence,
    random_unit_norm_matrix,
    split_into_windows,
)


class PositionalEncoding:
    @staticmethod
    def create_positional_encoder(
        pe_type: str = "absolute",
        pe_learnable: bool = True,
        pe_embedding_dim: int = 256,
        pe_max_sequence_length: int = 50,
        pe_dropout: float = 0.1,
        pe_init_scale: float = 1.0,
    ) -> nn.Module:
        """Create appropriate positional encoding based on parameters."""
        if pe_type == "none":
            return nn.Identity()
        elif pe_type == "absolute":
            return AbsolutePositionEncoding(
                d_model=pe_embedding_dim,
                dropout=pe_dropout,
                max_len=pe_max_sequence_length,
                learnable=pe_learnable,
                init_scale=pe_init_scale,
            )
        elif pe_type == "one_hot":
            return OneHotConcatPosition(embed_dim=pe_embedding_dim)
        else:
            return nn.Identity()


class ARModel(nn.Module, ABC):
    @classmethod
    def from_parameters(
        cls,
        dim: int,
        rank: int = 1,
        window: int = 1,
        scale: float = 1.0,
    ):
        raise NotImplementedError

    @abstractmethod
    def _get_weights(self):
        raise NotImplementedError

    @abstractmethod
    def _run_ar_model(
        self, x: torch.Tensor, weights: Union[torch.Tensor, nn.ModuleList]
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        tokens: torch.Tensor,
        unroll_sequences: bool = True,
        return_targets: bool = False,
        prefix: int = -1,
    ) -> torch.Tensor:
        # NOTE: pad becomes zero as prefix_length is set dynamically in runner.py to
        # be window * span_length
        # Use context_length if available (accounts for stride), otherwise fall back to sum
        context_len = getattr(self, 'context_length', sum(self.span_lengths))

        if unroll_sequences:
            n, seq_len, d = tokens.shape
            tokens, targets = split_into_windows(
                seq=tokens,
                window=context_len,
                pad=0,
            )
        else:
            tokens = pad_sequence(seq=tokens, pad=0)

        if isinstance(self, LinearARModel):
            logits = self._run_ar_model(
                tokens,
                self._get_weights(),
                prefix=prefix,
                span_lengths=self.span_lengths,
            )
        else:
            logits = self._run_ar_model(tokens, self._get_weights())

        if unroll_sequences:
            logits = logits.view(n, seq_len - context_len, d)
            targets = targets.view(n, seq_len - context_len, d)

        if return_targets:
            return logits, targets
        return logits


# === Linear AR ===
class LinearARModel(ARModel):
    def _run_ar_model(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        prefix: int,
        span_lengths: list,
    ) -> torch.Tensor:
        # Ensure window size matches the number of span lengths
        if hasattr(self, "window") and self.window != len(span_lengths):
            raise ValueError(
                f"Window size ({self.window}) must equal the number of span lengths ({len(span_lengths)})"
            )

        # Get stride (None means non-overlapping)
        stride = getattr(self, 'stride', None)

        # Determine which spans to process based on prefix
        if prefix > 0:
            # prefix represents number of lags/windows; calculate actual token count for variable spans
            if self.reverse_constants:
                weights = weights[:prefix, :, :]
                active_span_lengths = span_lengths[:prefix]
                if stride is not None:
                    # With stride: context_length = (num_windows - 1) * stride + last_span_length
                    prefix_token_count = (prefix - 1) * stride + active_span_lengths[-1]
                else:
                    prefix_token_count = sum(active_span_lengths)
                x = x[:, :prefix_token_count, :]
            else:
                weights = weights[-prefix:, :, :]
                active_span_lengths = span_lengths[-prefix:]
                if stride is not None:
                    prefix_token_count = (prefix - 1) * stride + active_span_lengths[-1]
                else:
                    prefix_token_count = sum(active_span_lengths)
                x = x[:, -prefix_token_count:, :]
        else:
            active_span_lengths = span_lengths

        # Group tokens into variable-length spans and sum over each span
        batch_size, total_tokens, dim = x.shape

        # Each window has a different span length from active_span_lengths
        num_windows = len(active_span_lengths)

        # Process each window with its corresponding span length
        x_aggregated_list = []

        # Get position weights if available
        position_weights = getattr(self, 'span_position_weights', None)

        for window_idx in range(num_windows):
            span_len = active_span_lengths[window_idx]

            # Calculate start position based on stride
            if stride is not None:
                start_idx = window_idx * stride
            else:
                # Non-overlapping: sum of previous span lengths
                start_idx = sum(active_span_lengths[:window_idx])

            end_idx = start_idx + span_len

            # Extract tokens for this window's span
            span_tokens = x[:, start_idx:end_idx, :]

            # Apply position weights if provided, otherwise sum with equal weights
            if position_weights is not None:
                # Create weight tensor: shape (1, span_len, 1) for broadcasting
                weights_tensor = torch.tensor(
                    position_weights,
                    device=span_tokens.device,
                    dtype=span_tokens.dtype
                ).view(1, span_len, 1)
                # Weighted sum over the span dimension
                span_aggregated = torch.sum(span_tokens * weights_tensor, dim=1)
            else:
                # Sum over the span dimension (equal weights)
                span_aggregated = torch.sum(span_tokens, dim=1)

            x_aggregated_list.append(span_aggregated)

        # Stack all windows: (batch, num_windows, dim)
        x_aggregated = torch.stack(x_aggregated_list, dim=1)

        return torch.squeeze(
            torch.sum(
                torch.matmul(
                    torch.unsqueeze(weights, dim=0),
                    torch.unsqueeze(x_aggregated, dim=-1),
                ),
                dim=1,
            ),
            dim=-1,
        )


class VectorARModel(LinearARModel):
    def __init__(self, params: torch.Tensor):
        super().__init__()
        self._params = nn.Parameter(data=params.detach().clone())

    @classmethod
    def from_parameters(
        cls,
        dim: int,
        span_lengths: list,
        rank: int = 1,
        window: int = 1,
        scale: float = 1.0,
        multiplicative_constant: float = 1.0,
        reverse_constants: bool = True,
        shared_matrix_across_lags: bool = False,
        orthogonal_matrices: bool = False,
        stride: int = None,
        span_position_weights: list = None,
    ):
        assert dim > 0, f"Dimension {dim} must be positive"
        assert rank > 0, f"Rank {rank} must be positive"
        assert window > 0, f"Window {window} must be positive"
        assert scale > 0, f"Scale {scale} must be positive"
        assert rank <= dim, f"Rank {rank} must be less than or equal to dim {dim}"

        # Validate and process stride
        if stride is not None:
            assert stride > 0, f"Stride {stride} must be positive"
            # Warn if stride > min span_length (gaps between intervals)
            min_span = min(span_lengths)
            if stride > min_span:
                import warnings
                warnings.warn(
                    f"Stride ({stride}) > min span_length ({min_span}) creates gaps between intervals"
                )
            # Compute context length with stride
            context_length = (len(span_lengths) - 1) * stride + span_lengths[-1]
        else:
            # Non-overlapping: context_length = sum of all spans
            context_length = sum(span_lengths)

        # Validate and normalize span_position_weights
        if span_position_weights is not None:
            # Check that all span_lengths are equal (required for shared weights)
            if len(set(span_lengths)) != 1:
                raise ValueError(
                    f"All span_lengths must be equal when using span_position_weights. "
                    f"Got: {span_lengths}"
                )

            span_len = span_lengths[0]

            # Check that weight length matches span length
            if len(span_position_weights) != span_len:
                raise ValueError(
                    f"span_position_weights length ({len(span_position_weights)}) "
                    f"must match span_length ({span_len})"
                )

            # Normalize weights to sum to 1.0
            weight_sum = sum(span_position_weights)
            if weight_sum <= 0:
                raise ValueError(
                    f"span_position_weights must sum to a positive value. "
                    f"Got sum: {weight_sum}"
                )
            span_position_weights = [w / weight_sum for w in span_position_weights]
        
        if shared_matrix_across_lags:
            # Use the same matrix for all lags
            base_matrix = random_unit_norm_matrix(dim, rank)
            matrices = [base_matrix] * window
        elif orthogonal_matrices:
            # Mutually Frobenius-orthogonal matrices: Tr(A_i^T A_j) = 0 for i ≠ j
            from src.utils import random_orthogonal_matrices
            matrices = random_orthogonal_matrices(window, dim, rank)
        else:
            # Use a different matrix for each lag
            matrices = [random_unit_norm_matrix(dim, rank) for _ in range(window)]

        # Calculate scaling constants per lag
        constants = [
            multiplicative_constant**i
            for i in (reversed(range(window)) if reverse_constants else range(window))
        ]

        # Scale each matrix by its constant and stack them into a single tensor
        A = torch.stack(
            [matrix * const for matrix, const in zip(matrices, constants)],
            dim=0,
        )
        A *= scale

        # Create and return the model instance
        model = cls(params=A)
        model.dim = dim
        model.rank = rank
        model.window = window
        model.scale = scale
        model.multiplicative_constant = multiplicative_constant
        model.reverse_constants = reverse_constants
        model.shared_matrix_across_lags = shared_matrix_across_lags
        model.orthogonal_matrices = orthogonal_matrices
        model.span_lengths = span_lengths
        model.stride = stride
        model.context_length = context_length
        model.span_position_weights = span_position_weights
        return model

    def _get_weights(self):
        return self._params
1

# === Transformer ===
def generate_square_subsequent_mask(sz, device):
    """Generate a mask to prevent attention to future positions."""
    mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        ff_hidden_layer,
        dropout,
        pe_type: str = "absolute",
        pe_max_sequence_length: int = 50,
        init_scale: float = 1.0,
        skip_connection: bool = True,
        layer_normalization: bool = True,
        use_query_projection: bool = True,
        use_key_projection: bool = True,
        use_value_projection: bool = True,
        use_output_projection: bool = True,
        use_mlp: bool = True,
        head_alphas: Optional[torch.Tensor] = None,
        attention_disentanglement: bool = False,
        teacher_readout: bool = False,
        teacher_matrices: Optional[torch.Tensor] = None,
        attention_bias: bool = True,
        value_init_scale: Optional[float] = None,
        query_init_scale: Optional[float] = None,
    ):
        super(DecoderBlock, self).__init__()

        self.init_scale = init_scale
        self.skip_connection = skip_connection
        self.layer_normalization = layer_normalization
        self.use_mlp = use_mlp
        self.attention_disentanglement = attention_disentanglement

        self.self_attention = MultiHeadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            pe_type=pe_type,
            pe_max_sequence_length=pe_max_sequence_length,
            init_scale=init_scale,
            use_query_projection=use_query_projection,
            use_key_projection=use_key_projection,
            use_value_projection=use_value_projection,
            use_output_projection=use_output_projection,
            head_alphas=head_alphas,
            attention_disentanglement=attention_disentanglement,
            teacher_readout=teacher_readout,
            teacher_matrices=teacher_matrices,
            attention_bias=attention_bias,
            value_init_scale=value_init_scale,
            query_init_scale=query_init_scale,
        )

        self.norm1 = (
            nn.LayerNorm(d_model) if self.layer_normalization else nn.Identity()
        )
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = (
            nn.Linear(d_model, ff_hidden_layer) if self.use_mlp else nn.Identity()
        )
        self.linear2 = (
            nn.Linear(ff_hidden_layer, d_model) if self.use_mlp else nn.Identity()
        )
        self.norm2 = (
            nn.LayerNorm(d_model) if self.layer_normalization else nn.Identity()
        )
        self.dropout2 = nn.Dropout(dropout)

        self._init_mlp_weights()

    def _init_mlp_weights(self):
        # Initialize MLP layers only
        for linear_layer in [self.linear1, self.linear2]:
            if isinstance(linear_layer, nn.Linear):
                nn.init.xavier_uniform_(linear_layer.weight)
                linear_layer.weight.data.mul_(self.init_scale)
                if linear_layer.bias is not None:
                    nn.init.zeros_(linear_layer.bias)

        # Initialize LayerNorm layers only
        for norm_layer in [self.norm1, self.norm2]:
            if isinstance(norm_layer, nn.LayerNorm):
                nn.init.ones_(norm_layer.weight)
                nn.init.zeros_(norm_layer.bias)

    def forward(self, x, target_mask):
        attn_output, attn_weights = self.self_attention(x, x, x, attn_mask=target_mask)
        x = (
            x + self.dropout1(attn_output)
            if self.skip_connection
            else self.dropout1(attn_output)
        )
        x = self.norm1(x)
        if self.use_mlp:
            ff_output = self.linear2(F.relu(self.linear1(x)))
            x = (
                x + self.dropout2(ff_output)
                if self.skip_connection
                else self.dropout2(ff_output)
            )
        x = self.norm2(x)
        return x, attn_weights


class OneHotConcatPosition(nn.Module):
    """
    Concatenate one-hot positional encoding.
    Input:  x [B, L, D]
    Output: [B, L, D + L]
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        # Shape: [1, embed_dim, embed_dim]
        eye = torch.eye(embed_dim, dtype=torch.float32).unsqueeze(0)
        self.register_buffer("onehot", eye, persistent=True)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        # NOTE: When training n-gram models: L < seq_len; Otherwise, L == seq_len.
        if L != self.embed_dim:
            raise ValueError(f"Expected embed_dim={self.embed_dim}, got {L}")
        pos = self.onehot.to(dtype=x.dtype, device=x.device).expand(B, L, L)
        return torch.cat([x, pos], dim=-1)  # [B, L, D + L]


class AbsolutePositionEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.0,
        max_len: int = 24,  # prefix + sequence length
        learnable: bool = True,
        init_scale: float = 1.0,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.learnable = learnable

        if self.learnable:
            self.pos_embedding = nn.Embedding(max_len, d_model)
            nn.init.xavier_uniform_(self.pos_embedding.weight)
            self.pos_embedding.weight.data.mul_(init_scale)
        else:
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model + 1, 2) * (-math.log(10000.0) / d_model)
            )
            pe = torch.zeros(1, max_len, d_model)
            pe[0, :, 0::2] = torch.sin(
                position * div_term[: d_model // 2 + d_model % 2]
            )
            pe[0, :, 1::2] = torch.cos(position * div_term[: d_model // 2])
            self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        seq_len = x.size(1)
        if self.learnable:
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
            x = x + self.pos_embedding(positions)
        else:
            x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class TeacherDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        dim: int,
        window: int,
        teacher_matrices: torch.Tensor,
    ):
        super().__init__()

        pad_total = hidden_dim - (window * dim)
        pad_each, remainder = divmod(pad_total, window)

        padded_blocks = []
        for k in range(window):
            teacher_matrix = teacher_matrices[k]  # (dim, dim)
            pad_cols = pad_each + (1 if k < remainder else 0)
            # F.pad format for 2D: (pad_left, pad_right)
            padded_teacher = F.pad(teacher_matrix, (0, pad_cols))
            padded_blocks.append(padded_teacher)

        # teacher_weights: [pad(A0) | pad(A1) | ... | pad(A_{W-1})]
        teacher_weights = torch.cat(padded_blocks, dim=1)  # (dim, hidden_dim)

        self.readout = nn.Linear(hidden_dim, dim, bias=False)
        with torch.no_grad():
            self.readout.weight.copy_(teacher_weights)
            self.readout.weight.requires_grad = False  # freeze parameters

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.readout(hidden)  # [batch, seq_len, dim]


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_heads: int,
        ff_hidden_dim: int,
        num_blocks: int,
        dropout: float = 0.1,
        pe_type: str = "absolute",
        pe_learnable: bool = True,
        pe_max_sequence_length: int = 50,
        pe_embedding_dim: int = -1,
        encoder_layer: bool = True,
        decoder_layer: bool = True,
        init_scale: float = 1.0,
        skip_connection: bool = True,
        layer_normalization: bool = True,
        use_query_projection: bool = True,
        use_key_projection: bool = True,
        use_value_projection: bool = True,
        use_output_projection: bool = True,
        use_mlp: bool = True,
        teacher_readout: bool = False,
        semantic_baseline: bool = False,
        attention_disentanglement: bool = False,
        identity_decoder: bool = False,
        attention_bias: bool = True,
        # Required when attention_disentanglement is True.
        window: Optional[int] = None,
        teacher_matrices: Optional[torch.Tensor] = None,
        # Required when use_semantic_baseline is True
        per_head_alpha: Optional[float] = None,
        # Custom initialization scales for attention weights
        value_init_scale: Optional[float] = None,
        query_init_scale: Optional[float] = None,
        teacher_target: bool = False,
    ):
        super(TransformerDecoder, self).__init__()

        assert dim > 0 and hidden_dim > 0, (
            f"Dimension {dim} and {hidden_dim} must be positive"
        )

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.ff_hidden_dim = ff_hidden_dim
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.encoder_layer = encoder_layer
        self.decoder_layer = decoder_layer
        self.init_scale = init_scale
        self.skip_connection = skip_connection
        self.layer_normalization = layer_normalization
        self.use_query_projection = use_query_projection
        self.use_key_projection = use_key_projection
        self.use_value_projection = use_value_projection
        self.use_output_projection = use_output_projection
        self.use_mlp = use_mlp
        self.teacher_readout = teacher_readout
        self.semantic_baseline = semantic_baseline
        self.attention_disentanglement = attention_disentanglement
        self.identity_decoder = identity_decoder
        self.per_head_alpha = per_head_alpha
        self.pe_type = pe_type
        self.pe_learnable = pe_learnable
        self.pe_max_sequence_length = pe_max_sequence_length
        self.pe_embedding_dim = pe_embedding_dim
        self.value_init_scale = value_init_scale
        self.query_init_scale = query_init_scale
        self.teacher_target = teacher_target

        # Ensure at least one of query or key projection is enabled for learning
        if not use_query_projection and not use_key_projection:
            raise ValueError(
                "At least one of use_query_projection or use_key_projection must be True for learning"
            )

        if semantic_baseline:
            assert per_head_alpha is not None and per_head_alpha > 0, (
                "When using semantic baseline, per_head_alpha must be provided and be a positive float."
            )

            if self.pe_type != "none":
                raise ValueError(
                    f"Semantic baseline requires pe_type='none', "
                    f"but got '{self.pe_type}'. Set pe_type='none'."
                )

            # Scale each head geometrically according to the given alpha.
            head_alphas = per_head_alpha ** torch.arange(num_heads, dtype=torch.float32)
        else:
            head_alphas = None

        if ff_hidden_dim == -1:
            ff_hidden_dim = hidden_dim

        # Set up encoder layer
        if self.pe_type == "one_hot":
            # One-hot encoding already given by the input.
            self.encoder = nn.Identity()
        else:
            self.encoder = (
                nn.Linear(dim, hidden_dim, bias=False)
                if (encoder_layer or (dim != hidden_dim))
                else nn.Identity()
            )
            if dim != hidden_dim:
                self.encoder.weight.requires_grad = encoder_layer
                nn.init.orthogonal_(self.encoder.weight)
                self.encoder.weight.data /= torch.linalg.norm(
                    self.encoder.weight.data.T, 2
                )
                self.encoder.weight.data.mul_(init_scale)

        # Create positional encoder
        self.pos_encoder = PositionalEncoding.create_positional_encoder(
            pe_type=self.pe_type,
            pe_learnable=self.pe_learnable,
            pe_embedding_dim=self.pe_embedding_dim,
            pe_max_sequence_length=self.pe_max_sequence_length,
            pe_dropout=self.dropout,
            pe_init_scale=self.init_scale,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    hidden_dim,
                    num_heads,
                    ff_hidden_dim,
                    dropout,
                    pe_type=self.pe_type,
                    pe_max_sequence_length=self.pe_max_sequence_length,
                    init_scale=init_scale,
                    skip_connection=skip_connection,
                    layer_normalization=layer_normalization,
                    use_query_projection=use_query_projection,
                    use_key_projection=use_key_projection,
                    use_value_projection=use_value_projection,
                    use_output_projection=use_output_projection,
                    use_mlp=use_mlp,
                    head_alphas=head_alphas,
                    attention_disentanglement=attention_disentanglement,
                    teacher_readout=teacher_readout,
                    teacher_matrices=teacher_matrices,
                    attention_bias=attention_bias,
                    value_init_scale=value_init_scale,
                    query_init_scale=query_init_scale,
                )
                for _ in range(num_blocks)
            ]
        )

        if teacher_readout and not attention_disentanglement:
            if identity_decoder:
                raise ValueError(
                    "Cannot use identity_decoder=True with teacher_readout=True when attention_disentanglement=False."
                )
            # fit teacher weights to decoder.
            self.decoder = TeacherDecoder(hidden_dim, dim, window, teacher_matrices)
        else:
            self.decoder = (
                nn.Linear(hidden_dim, dim, bias=False)
                if (decoder_layer or (dim != hidden_dim))
                else nn.Identity()
            )
            if dim != hidden_dim:
                # Extract the first `dim` components and ignore the rest.
                if identity_decoder:
                    identity_part = torch.eye(dim)
                    zero_part = torch.zeros(dim, hidden_dim - dim)
                    # Shape: (dim, hidden_dim)
                    decoder_weight = torch.cat([identity_part, zero_part], dim=1)
                    self.decoder.weight.data.copy_(decoder_weight)
                    self.decoder.weight.requires_grad = False
                else:
                    self.decoder.weight.requires_grad = decoder_layer
                    nn.init.orthogonal_(self.decoder.weight)
                    self.decoder.weight.data /= torch.linalg.norm(
                        self.decoder.weight.data.T, 2
                    )
                    self.decoder.weight.data.mul_(init_scale)

        # Freeze parameters when using semantic baseline.
        if semantic_baseline:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x):
        x = self.encoder(x)
        x = self.pos_encoder(x)
        target_mask = generate_square_subsequent_mask(x.size(1), device=x.device)
        attention_weights = []
        for transformer_block in self.transformer_blocks:
            x, attn_weights = transformer_block(x, target_mask)
            attention_weights.append(attn_weights)
        attention_weights = torch.stack(attention_weights)
        output = self.decoder(x)
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        pe_type: str = "absolute",
        pe_max_sequence_length: int = 50,
        init_scale: float = 1,
        use_query_projection: bool = True,
        use_key_projection: bool = True,
        use_output_projection: bool = True,
        use_value_projection: bool = True,
        head_alphas: Optional[torch.Tensor] = None,  # per-head scaling, shape [H]
        attention_disentanglement: bool = False,
        teacher_readout: bool = False,
        teacher_matrices: Optional[torch.Tensor] = None,
        attention_bias: bool = True,
        value_init_scale: Optional[float] = None,
        query_init_scale: Optional[float] = None,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention_disentanglement = attention_disentanglement
        self.teacher_readout = teacher_readout
        self.attention_bias = attention_bias
        self.value_init_scale = value_init_scale
        self.query_init_scale = query_init_scale

        if attention_disentanglement:
            self.head_dim = embed_dim  # all heads see the full embedding
        else:
            assert embed_dim % num_heads == 0, (
                "embed_dim must be divisible by num_heads"
            )
            self.head_dim = (
                embed_dim // num_heads
            )  # each head sees a portion of the embedding

        self.init_scale = init_scale

        # Store PE parameters
        self.pe_type = pe_type
        self.pe_max_sequence_length = pe_max_sequence_length

        if attention_disentanglement:
            # Create head-specific projections when attention_disentanglement=True
            # Each head gets its own set of parameters for truly independent parameters.
            # TODO: set bias to False in all projections?
            self.query_proj = (
                nn.ModuleList(
                    [
                        nn.Linear(embed_dim, embed_dim, bias=attention_bias)
                        for _ in range(num_heads)
                    ]
                )
                if use_query_projection
                else nn.ModuleList([nn.Identity() for _ in range(num_heads)])
            )
            self.key_proj = (
                nn.ModuleList(
                    [
                        nn.Linear(embed_dim, embed_dim, bias=attention_bias)
                        for _ in range(num_heads)
                    ]
                )
                if use_key_projection
                else nn.ModuleList([nn.Identity() for _ in range(num_heads)])
            )
            self.value_proj = (
                nn.ModuleList(
                    [
                        nn.Linear(embed_dim, embed_dim, bias=attention_bias)
                        for _ in range(num_heads)
                    ]
                )
                if use_value_projection or teacher_readout
                else nn.ModuleList([nn.Identity() for _ in range(num_heads)])
            )
            self.out_proj = (
                nn.ModuleList(
                    [
                        nn.Linear(embed_dim, embed_dim, bias=attention_bias)
                        for _ in range(num_heads)
                    ]
                )
                if use_output_projection
                else nn.ModuleList([nn.Identity() for _ in range(num_heads)])
            )
        else:
            # Standard shared projections
            self.query_proj = (
                nn.Linear(embed_dim, embed_dim, bias=attention_bias)
                if use_query_projection
                else nn.Identity()
            )
            self.key_proj = (
                nn.Linear(embed_dim, embed_dim, bias=attention_bias)
                if use_key_projection
                else nn.Identity()
            )
            self.value_proj = (
                nn.Linear(embed_dim, embed_dim, bias=attention_bias)
                if use_value_projection
                else nn.Identity()
            )
            self.out_proj = (
                nn.Linear(embed_dim, embed_dim, bias=attention_bias)
                if use_output_projection
                else nn.Identity()
            )

        # dropout on attention weights
        self.dropout = nn.Dropout(dropout)

        if head_alphas is not None:
            assert head_alphas.shape == (num_heads,), (
                f"head_alpha must be shape [{num_heads}]"
            )
            self.register_buffer(
                "head_alphas",
                head_alphas.view(1, num_heads, 1, 1),
                persistent=True,
            )
        else:
            self.head_alphas = None

        if teacher_readout and attention_disentanglement:
            if teacher_matrices is None:
                raise ValueError(
                    "teacher_matrices must be provided when teacher_readout is True"
                )
            self._init_teacher_value_weights(teacher_matrices)

        # Initialize weights
        self._init_attention_weights()

    def _init_teacher_value_weights(self, teacher_matrices):
        """Initialize value projection weights with teacher matrices and freeze them."""
        # teacher_matrices shape: (window, dim, dim)
        window, dim, _ = teacher_matrices.shape

        # Validate dimensions
        if window != self.num_heads:
            raise ValueError(
                f"Number of teacher matrices ({window}) must equal number of heads ({self.num_heads})"
            )

        with torch.no_grad():
            for head_idx in range(self.num_heads):
                teacher_matrix = teacher_matrices[head_idx]  # (dim, dim)
                # Pad teacher matrix to match embed_dim
                pad = self.embed_dim - dim
                if pad < 0:
                    raise ValueError(
                        f"embed_dim ({self.embed_dim}) must be at least as large as teacher matrix dimension ({dim})"
                    )
                # Pad to the right and bottom with zeroes.
                # F.pad format for 2D: (pad_left, pad_right, pad_top, pad_bottom)
                # TODO: why don't we make the padded elements learnable?
                padded_teacher_matrix = F.pad(
                    teacher_matrix, (0, pad, 0, pad)
                )  # (embed_dim, embed_dim)

                # Each head has its own value projection.
                # Copy the padded matrix to this head's value projection and freeze all parameters.
                self.value_proj[head_idx].weight.copy_(padded_teacher_matrix)
                self.value_proj[head_idx].weight.requires_grad = False

    def _init_attention_weights(self):
        if self.attention_disentanglement:
            for proj_list in (
                self.query_proj,
                self.key_proj,
                self.value_proj,
                self.out_proj,
            ):
                for i, proj in enumerate(proj_list):
                    if isinstance(proj, nn.Linear) and proj.weight.requires_grad:
                        # Apply custom initialization for query and value projections
                        if (
                            proj_list is self.query_proj
                            and self.query_init_scale is not None
                        ):
                            init.uniform_(
                                proj.weight,
                                -self.query_init_scale,
                                self.query_init_scale,
                            )
                        elif (
                            proj_list is self.value_proj
                            and self.value_init_scale is not None
                        ):
                            init.uniform_(
                                proj.weight,
                                -self.value_init_scale,
                                self.value_init_scale,
                            )
                        else:
                            init.xavier_uniform_(proj.weight)
                            proj.weight.data.mul_(self.init_scale)
                        if proj.bias is not None:
                            init.zeros_(proj.bias)
        else:
            for proj in (
                self.query_proj,
                self.key_proj,
                self.value_proj,
                self.out_proj,
            ):
                if isinstance(proj, nn.Linear):
                    # Apply custom initialization for query and value projections
                    if proj is self.query_proj and self.query_init_scale is not None:
                        init.uniform_(
                            proj.weight, -self.query_init_scale, self.query_init_scale
                        )
                    elif proj is self.value_proj and self.value_init_scale is not None:
                        init.uniform_(
                            proj.weight, -self.value_init_scale, self.value_init_scale
                        )
                    else:
                        init.xavier_uniform_(proj.weight)
                        proj.weight.data.mul_(self.init_scale)
                    if proj.bias is not None:
                        init.zeros_(proj.bias)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ):
        batch_size, seq_len, _ = query.size()
        device = query.device
        scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

        if self.attention_disentanglement:
            # Apply head-specific projections, each head gets different parameters
            # -> [batch, seq_len, num_heads, head_dim]
            r_q1_list = []
            r_k1_list = []
            r_v1_list = []

            for head_idx in range(self.num_heads):
                # Apply head-specific projections
                head_query = self.query_proj[head_idx](query)
                head_key = self.key_proj[head_idx](key)
                head_value = self.value_proj[head_idx](value)

                r_q1_list.append(
                    head_query.unsqueeze(2)
                )  # [batch, seq_len, 1, embed_dim]
                r_k1_list.append(head_key.unsqueeze(2))
                r_v1_list.append(head_value.unsqueeze(2))

            # Stack along head dimension: [batch, seq_len, num_heads, embed_dim]
            r_q1 = torch.cat(r_q1_list, dim=2)
            r_k1 = torch.cat(r_k1_list, dim=2)
            r_v1 = torch.cat(r_v1_list, dim=2)
        else:
            # Apply shared projections
            query = self.query_proj(query)
            key = self.key_proj(key)
            value = self.value_proj(value)

            # Split heads: [batch, seq_len, embed_dim] -> [batch, seq_len, heads, head_dim]
            r_q1 = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
            r_k1 = key.view(batch_size, seq_len, self.num_heads, self.head_dim)
            r_v1 = value.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose: [batch, seq_len, heads, head_dim] -> [batch, heads, seq_len, head_dim]
        r_q1 = r_q1.transpose(1, 2)
        r_k1 = r_k1.transpose(1, 2)
        r_v1 = r_v1.transpose(1, 2)

        # Loops implicitly over batch and heads. Output: [batch, head, seq_len, seq_len].
        attn = torch.matmul(r_q1, r_k1.transpose(-2, -1)) / scale

        # Apply mask
        if attn_mask is not None:
            attn = attn + attn_mask

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        attn_output = torch.matmul(attn, r_v1)  # [batch, heads, seq_len, head_dim]

        # Apply per-head scaling
        if self.head_alphas is not None:
            attn_output = (
                attn_output * self.head_alphas
            )  # [B,H,L,D] x [1,H,1,1] -> [B,H,L,D]

        if self.attention_disentanglement:
            # Attention output: [batch, heads, seq_len, output_dim]
            # Apply head-specific output projections then sum.
            head_outputs = []
            for head_idx in range(self.num_heads):
                # attn_output[:, head_idx, :, :] is [batch, seq_len, output_dim]
                head_out = self.out_proj[head_idx](attn_output[:, head_idx, :, :])
                head_outputs.append(
                    head_out.unsqueeze(1)
                )  # [batch, 1, seq_len, output_dim]
            output = torch.cat(
                head_outputs, dim=1
            )  # [batch, heads, seq_len, output_dim]
            output = torch.sum(
                output, dim=1
            )  # sum over heads -> [batch, seq_len, output_dim]
        else:
            # Concat heads & final projection
            attn_output = (
                attn_output.transpose(1, 2)
                .contiguous()
                .view(batch_size, seq_len, self.embed_dim)
            )
            output = self.out_proj(attn_output)

        return output, attn


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
