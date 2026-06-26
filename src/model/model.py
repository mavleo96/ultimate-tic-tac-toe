from dataclasses import dataclass

import torch
import torch.nn as nn

from .components import TransformerBlock


@dataclass
class SimpleTransformerEncoderConfig:
    n_layers: int = 4
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 256
    dropout: float = 0.1


class SimpleTransformerEncoder(nn.Module):
    def __init__(self, config: SimpleTransformerEncoderConfig) -> None:
        super().__init__()

        self.config = config

        # Embeddings
        # 1. board_emb: 3 possible board states (opponent 0, empty 1, self 2)
        # 2. pos_emb: 3x3 board positions flattened
        self.board_emb = nn.Embedding(3, config.d_model)
        self.pos_emb = nn.Embedding(9, config.d_model)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
                for _ in range(config.n_layers)
            ]
        )

        self.cls_token = nn.Parameter(torch.empty(config.d_model))

        self.norm = nn.LayerNorm(config.d_model)

        self.policy_head = nn.Linear(config.d_model, 1, bias=False)
        self.value_head = nn.Linear(config.d_model, 1, bias=False)

        self.register_buffer("positions", torch.arange(9))

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S = x.shape
        if S != 9:
            raise ValueError(f"Input must be a 9-cell board, got {S}")

        h = self.board_emb(x + 1) + self.pos_emb(self.positions)[None, :, :]
        h = torch.cat([self.cls_token[None, None, :].expand(B, -1, -1), h], dim=1)

        for block in self.blocks:
            h = block(h)

        h = self.norm(h)

        # policy: each position token predicts its own logit; value: CLS aggregates global state
        policy = self.policy_head(h[:, 1:, :]).squeeze(-1)
        value = self.value_head(h[:, 0, :]).squeeze(-1)

        value = torch.tanh(value)

        return policy, value
