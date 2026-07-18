from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.attention import MultiHeadAttention, generate_square_subsequent_mask
from src.model.positional import PositionalEncoding


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

    def predict_next(self, prefix: torch.Tensor) -> torch.Tensor:
        """Return next-token log-probs from the last position of the causal attention.

        prefix: (B, T, dim). Returns (B, dim) log-probs. Kept as a duck-typed
        method rather than participating in the ARTeacher hierarchy — the
        Predictor layer uses it. Matches the log-prob output convention of
        `ARTeacher.predict_next` so the predictor is teacher-type-agnostic.
        """
        logits, _ = self(prefix)
        return F.log_softmax(logits[:, -1, :], dim=-1)
