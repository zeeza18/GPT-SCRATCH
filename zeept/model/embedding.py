"""
Embedding layers for token and positional encodings.
"""

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Token embedding layer that converts token IDs to dense vectors.
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        """
        Initialize token embedding.

        Args:
            vocab_size: Size of the vocabulary
            embed_dim: Dimension of embedding vectors
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Token IDs tensor of shape (batch_size, seq_len)

        Returns:
            Embedded tokens of shape (batch_size, seq_len, embed_dim)
        """
        return self.embedding(x)


class PositionalEmbedding(nn.Module):
    """
    Learnable positional embedding layer.

    Adds position information to token embeddings.
    """

    def __init__(self, max_len: int, embed_dim: int):
        """
        Initialize positional embedding.

        Args:
            max_len: Maximum sequence length
            embed_dim: Dimension of embedding vectors
        """
        super().__init__()
        self.embedding = nn.Embedding(max_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            Positional embeddings of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        return self.embedding(positions)
