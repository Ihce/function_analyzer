import torch
import torch.nn as nn
from mamba_ssm import Mamba


class FunctionBoundaryModel(nn.Module):
    """Mamba-based model for function boundary detection."""

    def __init__(
        self,
        window_size: int = 4096,
        embed_dim: int = 256,
        mamba_dim: int = 512,
        n_layers: int = 4,
        dropout: float = 0.1,
        n_classes: int = 5,
    ):
        super().__init__()

        self.window_size = window_size
        self.n_classes = n_classes

        # Embeddings
        self.byte_embed = nn.Embedding(256, embed_dim)
        self.pos_embed = nn.Embedding(window_size, embed_dim)
        self.register_buffer("pos_indices", torch.arange(window_size))

        # Initial projection
        self.input_proj = nn.Linear(embed_dim, mamba_dim)

        # Mamba layers
        self.mamba_layers = nn.ModuleList(
            [
                Mamba(
                    d_model=mamba_dim,
                    d_state=64,
                    d_conv=4,
                    expand=2,
                    dt_rank="auto",
                    dt_min=0.001,
                    dt_max=0.1,
                    dt_init="random",
                    dt_scale=1.0,
                    dt_init_floor=1e-4,
                )
                for _ in range(n_layers)
            ]
        )

        self.layer_norms = nn.ModuleList([nn.LayerNorm(mamba_dim) for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(mamba_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, byte_sequence):
        """Forward pass."""
        B, L = byte_sequence.shape

        # Embeddings
        byte_emb = self.byte_embed(byte_sequence)
        pos_emb = self.pos_embed(self.pos_indices[:L].unsqueeze(0).expand(B, -1))
        x = byte_emb + pos_emb

        # Project to Mamba dimension
        x = self.input_proj(x)

        # Mamba layers with residual connections
        for i, (mamba, ln) in enumerate(zip(self.mamba_layers, self.layer_norms)):
            residual = x
            x = ln(x)
            x = mamba(x)
            x = self.dropout(x)
            x = x + residual

        # Output predictions
        logits = self.output_head(x)

        return logits  # Shape: [B, L, n_classes]
