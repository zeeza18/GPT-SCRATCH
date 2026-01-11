"""
Tokenizer implementations for LLM text processing.
Includes simple regex-based tokenizer and BPE tokenizer support.
"""

import re
from typing import List, Dict, Optional
import tiktoken


class SimpleTokenizer:
    """
    Simple regex-based tokenizer for text preprocessing.

    Splits text on whitespace and punctuation while preserving special characters.
    Useful for basic tokenization tasks and prototyping.
    """

    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        """
        Initialize the SimpleTokenizer.

        Args:
            vocab: Optional pre-built vocabulary dictionary mapping tokens to IDs
        """
        self.vocab = vocab or {}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into a list of tokens.

        Args:
            text: Input text string

        Returns:
            List of token strings
        """
        # Split on whitespace and common punctuation
        result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        # Remove empty tokens and strip whitespace
        tokens = [item.strip() for item in result if item.strip()]
        return tokens

    def build_vocab(self, text: str) -> None:
        """
        Build vocabulary from text.

        Args:
            text: Input text to build vocabulary from
        """
        tokens = self.tokenize(text)
        unique_tokens = sorted(set(tokens))
        self.vocab = {token: idx for idx, token in enumerate(unique_tokens)}
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}

    def encode(self, text: str) -> List[int]:
        """
        Encode text into token IDs.

        Args:
            text: Input text string

        Returns:
            List of token IDs
        """
        tokens = self.tokenize(text)
        return [self.vocab.get(token, 0) for token in tokens]

    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs back into text.

        Args:
            ids: List of token IDs

        Returns:
            Decoded text string
        """
        tokens = [self.inverse_vocab.get(id, "<UNK>") for id in ids]
        return " ".join(tokens)

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)


class BPETokenizer:
    """
    Byte Pair Encoding (BPE) tokenizer using tiktoken.

    Provides efficient tokenization for LLMs using the GPT-2 BPE algorithm.
    """

    def __init__(self, model_name: str = "gpt2"):
        """
        Initialize BPE tokenizer.

        Args:
            model_name: Name of the tiktoken model (default: gpt2)
        """
        self.tokenizer = tiktoken.get_encoding(model_name)
        self.model_name = model_name

    def encode(self, text: str) -> List[int]:
        """
        Encode text into BPE token IDs.

        Args:
            text: Input text string

        Returns:
            List of token IDs
        """
        return self.tokenizer.encode(text, allowed_special="all")

    def decode(self, ids: List[int]) -> str:
        """
        Decode BPE token IDs back into text.

        Args:
            ids: List of token IDs

        Returns:
            Decoded text string
        """
        return self.tokenizer.decode(ids)

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.tokenizer.n_vocab

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into BPE tokens.

        Args:
            text: Input text string

        Returns:
            List of token strings
        """
        ids = self.encode(text)
        return [self.tokenizer.decode_single_token_bytes(id).decode('utf-8', errors='replace')
                for id in ids]
