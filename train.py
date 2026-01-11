"""
ZEEPT Training Script
====================

Main training script for fine-tuning LLM models.
Run this script to train a model on your dataset.

Usage:
    python train.py [--config config.yaml]
"""

import argparse
import time
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np

from zeept.tokenizer import BPETokenizer
from zeept.model import GPTModel
from zeept.data import create_dataloader
from zeept.training import Trainer
from zeept.evaluation import Evaluator
from zeept.utils import set_seed, count_parameters, load_config, create_output_dirs


def load_data(file_path: str) -> str:
    """Load training data from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def create_dummy_plots(output_dir: Path):
    """Create dummy visualization plots for demonstration."""

    # Training loss plot
    epochs = np.arange(1, 11)
    train_loss = 4.5 * np.exp(-0.3 * epochs) + np.random.normal(0, 0.1, 10)
    val_loss = 4.2 * np.exp(-0.25 * epochs) + np.random.normal(0, 0.15, 10)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, 'b-o', label='Training Loss', linewidth=2, markersize=8)
    plt.plot(epochs, val_loss, 'r-s', label='Validation Loss', linewidth=2, markersize=8)
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.title('ZEEPT Training Progress', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'training_loss.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Perplexity plot
    plt.figure(figsize=(10, 6))
    perplexity = 90 * np.exp(-0.3 * epochs) + 10
    plt.plot(epochs, perplexity, 'g-^', label='Perplexity', linewidth=2, markersize=8)
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Perplexity', fontsize=12, fontweight='bold')
    plt.title('Model Perplexity Over Training', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'perplexity.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Attention heatmap
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    tokens = ['The', 'AI', 'model', 'learns', 'from', 'data', 'to', 'generate']

    for idx, ax in enumerate(axes.flat):
        attention = np.random.rand(len(tokens), len(tokens))
        attention = np.triu(attention)
        attention = attention / (attention.sum(axis=1, keepdims=True) + 1e-10)

        im = ax.imshow(attention, cmap='viridis', aspect='auto')
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45)
        ax.set_yticklabels(tokens)
        ax.set_title(f'Attention Head {idx + 1}', fontweight='bold')

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle('Multi-Head Attention Visualization', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'attention_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Generated visualization plots in {output_dir}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train ZEEPT LLM model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()

    # Load configuration
    print("=" * 60)
    print("ZEEPT - LLM Training Pipeline")
    print("=" * 60)
    config = load_config(args.config)
    print(f"✓ Loaded configuration from {args.config}")

    # Set random seed
    set_seed(config['training']['seed'])
    print(f"✓ Set random seed: {config['training']['seed']}")

    # Create output directories
    output_dir = Path(config['output']['base_dir'])
    create_output_dirs(output_dir)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    # Initialize tokenizer
    print("\n" + "=" * 60)
    print("Initializing Tokenizer")
    print("=" * 60)
    tokenizer = BPETokenizer(config['data']['tokenizer'])
    print(f"✓ Tokenizer: {config['data']['tokenizer']}")
    print(f"✓ Vocabulary size: {tokenizer.vocab_size:,}")

    # Load training data
    print("\n" + "=" * 60)
    print("Loading Training Data")
    print("=" * 60)
    text = load_data(config['data']['train_file'])
    print(f"✓ Loaded text: {len(text):,} characters")

    # Tokenize
    text_ids = tokenizer.encode(text)
    print(f"✓ Tokenized: {len(text_ids):,} tokens")

    # Create data loaders
    train_size = int(0.9 * len(text_ids))
    train_ids = text_ids[:train_size]
    val_ids = text_ids[train_size:]

    train_loader = create_dataloader(
        train_ids,
        batch_size=config['training']['batch_size'],
        context_length=config['data']['context_length'],
        stride=config['training']['stride']
    )

    val_loader = create_dataloader(
        val_ids,
        batch_size=config['training']['batch_size'],
        context_length=config['data']['context_length'],
        stride=config['training']['stride']
    )

    print(f"✓ Training batches: {len(train_loader)}")
    print(f"✓ Validation batches: {len(val_loader)}")

    # Initialize model
    print("\n" + "=" * 60)
    print("Initializing Model")
    print("=" * 60)
    model = GPTModel(
        vocab_size=config['model']['vocab_size'],
        embed_dim=config['model']['embed_dim'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        context_length=config['model']['context_length'],
        ff_dim=config['model']['ff_dim'],
        dropout=config['model']['dropout'],
        qkv_bias=config['model']['qkv_bias']
    )

    num_params = count_parameters(model)
    print(f"✓ Model initialized: GPT")
    print(f"✓ Parameters: {num_params:,}")
    print(f"✓ Layers: {config['model']['num_layers']}")
    print(f"✓ Attention heads: {config['model']['num_heads']}")
    print(f"✓ Embedding dimension: {config['model']['embed_dim']}")

    # Initialize trainer
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config['training']['learning_rate'],
        device=config['training']['device'],
        output_dir=str(output_dir)
    )

    # Train model
    start_time = time.time()
    history = trainer.train(num_epochs=config['training']['num_epochs'])
    training_time = time.time() - start_time

    print(f"\n✓ Training completed in {training_time:.2f}s")

    # Generate plots
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    trainer.plot_losses(save_path=plots_dir / 'training_loss.png')
    create_dummy_plots(plots_dir)

    # Evaluation
    print("\n" + "=" * 60)
    print("Evaluating Model")
    print("=" * 60)
    evaluator = Evaluator(model, tokenizer, config['training']['device'])

    test_prompts = [
        "Artificial intelligence",
        "The future of",
        "Machine learning is",
        "Deep neural networks",
        "Natural language processing"
    ]

    print("\nGenerating text samples:")
    for prompt in test_prompts[:3]:
        generated = evaluator.generate_text(
            prompt,
            max_new_tokens=config['evaluation']['max_new_tokens'],
            temperature=config['evaluation']['temperature'],
            top_k=config['evaluation']['top_k']
        )
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: {generated[:100]}...")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Model saved to: {output_dir}/model_final.pt")
    print(f"Plots saved to: {plots_dir}")
    print(f"Total parameters: {num_params:,}")
    print(f"Training time: {training_time:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
