"""
Training pipeline for LLM fine-tuning.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from typing import Optional, Dict, List
import matplotlib.pyplot as plt
from pathlib import Path
import json


class Trainer:
    """
    Trainer class for LLM fine-tuning.

    Handles training loop, optimization, and metric tracking.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 5e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "outputs"
    ):
        """
        Initialize trainer.

        Args:
            model: The model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            learning_rate: Learning rate for optimizer
            device: Device to train on ('cuda' or 'cpu')
            output_dir: Directory to save outputs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        self.train_losses = []
        self.val_losses = []
        self.tokens_seen = []

    def train_epoch(self) -> float:
        """
        Train for one epoch.

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)

            # Forward pass
            logits = self.model(input_ids)
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1)
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def validate(self) -> float:
        """
        Validate the model.

        Returns:
            Average validation loss
        """
        if self.val_loader is None:
            return 0.0

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)

                logits = self.model(input_ids)
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    target_ids.view(-1)
                )

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def train(self, num_epochs: int) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.

        Args:
            num_epochs: Number of epochs to train

        Returns:
            Dictionary containing training history
        """
        print(f"Training on {self.device}...")
        print(f"Total epochs: {num_epochs}")

        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            val_loss = self.validate()
            self.val_losses.append(val_loss)

            print(f"Epoch {epoch + 1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch + 1)

        # Save final model
        self.save_model()

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    def save_model(self):
        """Save final trained model."""
        model_path = self.output_dir / "model_final.pt"
        torch.save(self.model.state_dict(), model_path)
        print(f"Saved final model: {model_path}")

    def plot_losses(self, save_path: Optional[str] = None):
        """
        Plot training and validation losses.

        Args:
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.train_losses) + 1)

        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        if self.val_losses and any(v > 0 for v in self.val_losses):
            plt.plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)

        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved loss plot: {save_path}")
        else:
            plt.savefig(self.output_dir / "training_loss.png", dpi=300, bbox_inches='tight')

        plt.close()
