"""Mismo transformer sencillo (encoder-only) pero usando los módulos built-in de PyTorch."""

import torch
import torch.nn as nn


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 512,
        n_layers: int = 4,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)

        x = self.embedding(x) + self.pos_encoding(positions)
        x = self.encoder(x, mask=mask)
        x = self.norm(x)
        return self.output_proj(x)


if __name__ == "__main__":
    vocab_size = 1000
    model = SimpleTransformer(vocab_size=vocab_size)

    batch_size, seq_len = 2, 16
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))

    logits = model(tokens)
    print("Input shape:", tokens.shape)
    print("Output shape:", logits.shape)
