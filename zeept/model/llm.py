"""
Complete LLM model architecture using transformer blocks.
"""

import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .embedding import TokenEmbedding, PositionalEmbedding


class FeedForward(nn.Module):
    """
    Feed-forward network used in transformer blocks.
    """

    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.1):
        """
        Initialize feed-forward network.

        Args:
            embed_dim: Embedding dimension
            ff_dim: Hidden dimension of feed-forward network
            dropout: Dropout probability
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.layers(x)


class TransformerBlock(nn.Module):
    """
    Single transformer block with multi-head attention and feed-forward network.
    """

    def __init__(self, embed_dim: int, num_heads: int, context_length: int,
                 ff_dim: int, dropout: float = 0.1, qkv_bias: bool = False):
        """
        Initialize transformer block.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            context_length: Maximum context length
            ff_dim: Feed-forward hidden dimension
            dropout: Dropout probability
            qkv_bias: Whether to use bias in attention projections
        """
        super().__init__()
        self.attention = MultiHeadAttention(
            d_in=embed_dim,
            d_out=embed_dim,
            context_length=context_length,
            num_heads=num_heads,
            dropout=dropout,
            qkv_bias=qkv_bias
        )
        self.ff = FeedForward(embed_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connections.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        # Multi-head attention with residual connection
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)

        # Feed-forward with residual connection
        ff_out = self.ff(self.norm2(x))
        x = x + ff_out

        return x


class GPTModel(nn.Module):
    """
    GPT-style language model with transformer architecture.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 768, num_heads: int = 12,
                 num_layers: int = 12, context_length: int = 1024,
                 ff_dim: int = 3072, dropout: float = 0.1, qkv_bias: bool = False):
        """
        Initialize GPT model.

        Args:
            vocab_size: Size of vocabulary
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            context_length: Maximum context length
            ff_dim: Feed-forward hidden dimension
            dropout: Dropout probability
            qkv_bias: Whether to use bias in attention
        """
        super().__init__()
        self.token_emb = TokenEmbedding(vocab_size, embed_dim)
        self.pos_emb = PositionalEmbedding(context_length, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                context_length=context_length,
                ff_dim=ff_dim,
                dropout=dropout,
                qkv_bias=qkv_bias
            ) for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embed_dim)
        self.out_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input token IDs of shape (batch_size, seq_len)

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        # Embedding
        token_embeds = self.token_emb(x)
        pos_embeds = self.pos_emb(token_embeds)
        x = self.dropout(token_embeds + pos_embeds)

        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Output projection
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits
