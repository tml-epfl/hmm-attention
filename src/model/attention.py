from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def generate_square_subsequent_mask(sz, device):
    """Generate a mask to prevent attention to future positions."""
    mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


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
