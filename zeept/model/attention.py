"""
Attention mechanism implementations for transformer-based LLMs.
Includes self-attention and multi-head attention.
"""

import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):
    """
    Self-Attention mechanism for transformers.

    Computes attention scores between all positions in a sequence
    and produces context-aware representations.
    """

    def __init__(self, d_in: int, d_out: int, qkv_bias: bool = False):
        """
        Initialize self-attention layer.

        Args:
            d_in: Input dimension
            d_out: Output dimension
            qkv_bias: Whether to use bias in query, key, value projections
        """
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_in)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_out)
        """
        # Project input to queries, keys, and values
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # Compute attention scores
        attn_scores = queries @ keys.transpose(-2, -1)

        # Scale by square root of dimension
        attn_scores = attn_scores / math.sqrt(self.d_out)

        # Apply softmax to get attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Compute weighted sum of values
        context_vec = attn_weights @ values

        return context_vec


class CausalSelfAttention(SelfAttention):
    """
    Causal (masked) self-attention for autoregressive generation.

    Prevents attention to future positions in the sequence.
    """

    def __init__(self, d_in: int, d_out: int, context_length: int,
                 dropout: float = 0.0, qkv_bias: bool = False):
        """
        Initialize causal self-attention.

        Args:
            d_in: Input dimension
            d_out: Output dimension
            context_length: Maximum sequence length
            dropout: Dropout probability
            qkv_bias: Whether to use bias in QKV projections
        """
        super().__init__(d_in, d_out, qkv_bias)
        self.dropout = nn.Dropout(dropout)
        # Create causal mask
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with causal masking.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_in)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_out)
        """
        b, num_tokens, d_in = x.shape

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(-2, -1)
        attn_scores = attn_scores / math.sqrt(self.d_out)

        # Apply causal mask
        attn_scores = attn_scores.masked_fill(
            self.mask[:num_tokens, :num_tokens].bool(), float('-inf')
        )

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values

        return context_vec


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.

    Runs multiple attention heads in parallel and combines their outputs.
    """

    def __init__(self, d_in: int, d_out: int, context_length: int,
                 num_heads: int, dropout: float = 0.0, qkv_bias: bool = False):
        """
        Initialize multi-head attention.

        Args:
            d_in: Input dimension
            d_out: Output dimension
            context_length: Maximum sequence length
            num_heads: Number of attention heads
            dropout: Dropout probability
            qkv_bias: Whether to use bias in QKV projections
        """
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        # Causal mask
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of multi-head attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_in)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_out)
        """
        b, num_tokens, d_in = x.shape

        # Project and split into multiple heads
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # Reshape: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute attention scores
        attn_scores = queries @ keys.transpose(-2, -1)
        attn_scores = attn_scores / math.sqrt(self.head_dim)

        # Apply causal mask
        attn_scores = attn_scores.masked_fill(
            self.mask[:num_tokens, :num_tokens].bool(), float('-inf')
        )

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute context vectors
        context_vec = attn_weights @ values

        # Reshape back: (b, num_heads, num_tokens, head_dim) -> (b, num_tokens, d_out)
        context_vec = context_vec.transpose(1, 2).contiguous()
        context_vec = context_vec.view(b, num_tokens, self.d_out)

        # Final projection
        context_vec = self.out_proj(context_vec)

        return context_vec
