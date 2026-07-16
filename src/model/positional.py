import math

import torch
import torch.nn as nn


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
