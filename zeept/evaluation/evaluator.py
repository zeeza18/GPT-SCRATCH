"""
Model evaluation and text generation utilities.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class Evaluator:
    """
    Evaluator for LLM model inference and evaluation.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize evaluator.

        Args:
            model: The model to evaluate
            tokenizer: Tokenizer for encoding/decoding
            device: Device to run on
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter

        Returns:
            Generated text
        """
        self.model.eval()

        # Encode prompt
        input_ids = torch.tensor(
            self.tokenizer.encode(prompt),
            dtype=torch.long
        ).unsqueeze(0).to(self.device)

        # Generate
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                logits = self.model(input_ids)

                # Get logits for last token
                logits = logits[:, -1, :] / temperature

                # Apply top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')

                # Sample from distribution
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)

        # Decode
        generated_ids = input_ids[0].tolist()
        generated_text = self.tokenizer.decode(generated_ids)

        return generated_text

    def calculate_perplexity(self, text: str) -> float:
        """
        Calculate perplexity on given text.

        Args:
            text: Input text

        Returns:
            Perplexity score
        """
        self.model.eval()

        # Encode text
        input_ids = torch.tensor(
            self.tokenizer.encode(text),
            dtype=torch.long
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids)

            # Calculate cross-entropy loss
            criterion = nn.CrossEntropyLoss()
            loss = criterion(
                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                input_ids[:, 1:].contiguous().view(-1)
            )

            perplexity = torch.exp(loss).item()

        return perplexity

    def evaluate_on_prompts(
        self,
        prompts: List[str],
        max_new_tokens: int = 50
    ) -> Dict[str, List]:
        """
        Evaluate model on multiple prompts.

        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens to generate

        Returns:
            Dictionary with prompts and generated texts
        """
        results = {
            'prompts': [],
            'generated': [],
            'scores': []
        }

        for prompt in prompts:
            generated = self.generate_text(prompt, max_new_tokens)
            score = self.calculate_perplexity(generated)

            results['prompts'].append(prompt)
            results['generated'].append(generated)
            results['scores'].append(score)

        return results

    def plot_attention_weights(
        self,
        text: str,
        layer_idx: int = 0,
        head_idx: int = 0,
        save_path: Optional[str] = None
    ):
        """
        Visualize attention weights.

        Args:
            text: Input text
            layer_idx: Layer index to visualize
            head_idx: Attention head index
            save_path: Path to save plot
        """
        # This is a simplified dummy visualization
        # In production, you'd extract actual attention weights from the model

        tokens = self.tokenizer.tokenize(text)[:20]  # Limit for visualization
        n_tokens = len(tokens)

        # Create dummy attention matrix
        attention = np.random.rand(n_tokens, n_tokens)
        # Make it upper triangular (causal)
        attention = np.triu(attention)
        # Normalize
        attention = attention / attention.sum(axis=1, keepdims=True)

        plt.figure(figsize=(10, 8))
        plt.imshow(attention, cmap='viridis', aspect='auto')
        plt.colorbar(label='Attention Weight')
        plt.xticks(range(n_tokens), tokens, rotation=90)
        plt.yticks(range(n_tokens), tokens)
        plt.xlabel('Key Tokens')
        plt.ylabel('Query Tokens')
        plt.title(f'Attention Weights - Layer {layer_idx}, Head {head_idx}')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
