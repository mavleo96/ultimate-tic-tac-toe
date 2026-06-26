from dataclasses import dataclass

import torch
import torch.nn as nn

from .components import TransformerBlock


@dataclass
class HierarchicalTransformerEncoderConfig:
    n_local_layers: int = 4
    n_meta_layers: int = 4
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 256
    dropout: float = 0.1


class HierarchicalTransformerEncoder(nn.Module):
    def __init__(self, config: HierarchicalTransformerEncoderConfig) -> None:
        super().__init__()

        self.config = config

        # Embeddings
        # 1. board_emb: 3 possible board states (opponent 0, empty 1, self 2)
        # 2. active_board_emb: binary flag (inactive 0, active 1) per board position
        # 3. local_pos_emb: 3x3 board positions flattened
        # 4. meta_pos_emb: 3x3 board positions flattened
        self.board_emb = nn.Embedding(3, config.d_model)
        self.active_board_emb = nn.Embedding(2, config.d_model)
        self.local_pos_emb = nn.Embedding(9, config.d_model)
        self.meta_pos_emb = nn.Embedding(9, config.d_model)

        # Transformer blocks
        # 1. local_blocks: process local boards independently
        # 2. meta_blocks: process meta board only
        self.local_blocks = nn.ModuleList(
            [
                TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
                for _ in range(config.n_local_layers)
            ]
        )
        self.meta_blocks = nn.ModuleList(
            [
                TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
                for _ in range(config.n_meta_layers)
            ]
        )

        # Cls tokens
        # 1. cls_local_token: aggregates info from each local board
        # 2. cls_meta_token: aggregates info from the meta board
        self.cls_local_token = nn.Parameter(torch.empty(config.d_model))
        self.cls_meta_token = nn.Parameter(torch.empty(config.d_model))

        # Normalization
        # 1. norm_local: normalizes local board cls embeddings
        # 2. norm: final normalization of meta embeddings
        self.norm_local = nn.LayerNorm(config.d_model)
        self.norm = nn.LayerNorm(config.d_model)

        self.policy_head = nn.Linear(config.d_model, 9, bias=False)
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

        nn.init.normal_(self.cls_local_token, std=0.02)
        nn.init.normal_(self.cls_meta_token, std=0.02)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        B, N, S = x.shape
        D = self.config.d_model
        if N != 9 or S != 9:
            raise ValueError(f"Input must be a 9x9 board (got {N}x{S})")

        h = self.board_emb(x + 1) + self.local_pos_emb(self.positions)[None, None, :, :]
        h = torch.cat(
            [self.cls_local_token[None, None, None, :].expand(B, 9, -1, -1), h],
            dim=2,
        )

        h = h.reshape(B * 9, 10, D)
        for block in self.local_blocks:
            h = block(h)
        h = h.reshape(B, 9, 10, D)

        h_meta = self.norm_local(h[:, :, 0, :])
        h_meta = (
            h_meta + self.meta_pos_emb(self.positions)[None, :, :] + self.active_board_emb(y.long())
        )
        h_meta = torch.cat(
            [self.cls_meta_token[None, None, :].expand(B, -1, -1), h_meta],
            dim=1,
        )

        for block in self.meta_blocks:
            h_meta = block(h_meta)

        h_meta = self.norm(h_meta)

        policy = self.policy_head(h_meta[:, 1:, :])
        value = self.value_head(h_meta[:, 0, :])

        policy = policy.reshape(B, -1)
        value = torch.tanh(value).squeeze(-1)

        return policy, value
