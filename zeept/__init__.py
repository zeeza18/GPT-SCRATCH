"""
ZEEPT - Advanced LLM Fine-tuning and Evaluation Platform
==========================================================

A modular framework for training, fine-tuning, and evaluating large language models
with state-of-the-art tokenization, attention mechanisms, and evaluation metrics.

Modules:
    - tokenizer: BPE and custom tokenization implementations
    - model: LLM architecture components (embeddings, attention, transformer blocks)
    - training: Training pipeline and optimization
    - evaluation: Model evaluation and metrics
    - data: Dataset handling and preprocessing
    - utils: Helper functions and utilities
"""

__version__ = "1.0.0"
__author__ = "ZEEPT Team"
__license__ = "MIT"

from zeept.tokenizer import SimpleTokenizer, BPETokenizer
from zeept.model import SelfAttention, MultiHeadAttention, TransformerBlock
from zeept.training import Trainer
from zeept.evaluation import Evaluator

__all__ = [
    "SimpleTokenizer",
    "BPETokenizer",
    "SelfAttention",
    "MultiHeadAttention",
    "TransformerBlock",
    "Trainer",
    "Evaluator",
]
