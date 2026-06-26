import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, Tq, _ = q.shape
        _, Tk, _ = k.shape

        Q = self.q_proj(q).reshape(B, Tq, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(k).reshape(B, Tk, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(v).reshape(B, Tk, self.n_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().reshape(B, Tq, -1)

        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm transformer block (self-attention only).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.BoolTensor | None = None,
    ) -> torch.Tensor:

        residual = x
        x = self.norm1(x)
        x = residual + self.attn(x, x, x, attn_mask)

        residual = x
        x = self.norm2(x)
        x = residual + self.mlp(x)

        return x


class MultiHeadAttentionPooling(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_seeds: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_seeds = n_seeds
        self.dropout = dropout

        # Learnable pooling queries
        self.seeds = nn.Parameter(torch.empty(n_seeds, d_model))

        self.mha = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        nn.init.normal_(self.seeds, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        B, _, D = x.shape

        if D != self.d_model:
            raise ValueError(f"x last dim ({D}) must match d_model ({self.d_model})")

        # Learnable queries broadcast across batch
        q = self.seeds.unsqueeze(0).expand(B, -1, -1)

        out = self.mha(q, x, x)

        if self.n_seeds == 1:
            out = out.squeeze(1)

        return out
