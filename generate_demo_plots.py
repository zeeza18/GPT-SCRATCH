"""
Generate dummy plots and visualizations for README screenshots
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Create assets directory
assets_dir = Path("assets")
assets_dir.mkdir(exist_ok=True)

# Set style
plt.style.use('dark_background')

# 1. Training Loss Plot
fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1e293b')
ax.set_facecolor('#0f172a')

epochs = np.arange(1, 11)
train_loss = 4.5 * np.exp(-0.3 * epochs) + np.random.normal(0, 0.08, 10)
val_loss = 4.2 * np.exp(-0.25 * epochs) + np.random.normal(0, 0.12, 10)

ax.plot(epochs, train_loss, 'o-', color='#6366f1', linewidth=3, markersize=10, label='Training Loss', markeredgewidth=2, markeredgecolor='white')
ax.plot(epochs, val_loss, 's-', color='#8b5cf6', linewidth=3, markersize=10, label='Validation Loss', markeredgewidth=2, markeredgecolor='white')

ax.set_xlabel('Epoch', fontsize=14, fontweight='bold', color='#f1f5f9')
ax.set_ylabel('Loss', fontsize=14, fontweight='bold', color='#f1f5f9')
ax.set_title('ZEEPT Training Progress', fontsize=18, fontweight='bold', color='#f1f5f9', pad=20)
ax.legend(fontsize=12, loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.2, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(assets_dir / 'training_loss.png', dpi=300, bbox_inches='tight', facecolor='#1e293b')
plt.close()

# 2. Perplexity Plot
fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1e293b')
ax.set_facecolor('#0f172a')

perplexity = 90 * np.exp(-0.3 * epochs) + 10 + np.random.normal(0, 1.5, 10)

ax.plot(epochs, perplexity, '^-', color='#10b981', linewidth=3, markersize=10, markeredgewidth=2, markeredgecolor='white')
ax.fill_between(epochs, perplexity - 2, perplexity + 2, alpha=0.2, color='#10b981')

ax.set_xlabel('Epoch', fontsize=14, fontweight='bold', color='#f1f5f9')
ax.set_ylabel('Perplexity', fontsize=14, fontweight='bold', color='#f1f5f9')
ax.set_title('Model Perplexity Over Training', fontsize=18, fontweight='bold', color='#f1f5f9', pad=20)
ax.grid(True, alpha=0.2, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(assets_dir / 'perplexity.png', dpi=300, bbox_inches='tight', facecolor='#1e293b')
plt.close()

# 3. Attention Heatmap
fig, axes = plt.subplots(2, 2, figsize=(14, 12), facecolor='#1e293b')
tokens = ['The', 'AI', 'model', 'learns', 'from', 'data', 'to', 'generate']

for idx, ax in enumerate(axes.flat):
    ax.set_facecolor('#0f172a')

    # Create realistic attention pattern
    attention = np.random.rand(len(tokens), len(tokens))
    attention = np.triu(attention)  # Causal mask
    # Add diagonal emphasis
    for i in range(len(tokens)):
        attention[i, max(0, i-2):i+1] += 0.5
    attention = attention / (attention.sum(axis=1, keepdims=True) + 1e-10)

    im = ax.imshow(attention, cmap='viridis', aspect='auto', interpolation='nearest')
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right', color='#f1f5f9')
    ax.set_yticklabels(tokens, color='#f1f5f9')
    ax.set_title(f'Attention Head {idx + 1}', fontweight='bold', color='#f1f5f9', fontsize=12)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors='#f1f5f9')

plt.suptitle('Multi-Head Attention Visualization', fontsize=18, fontweight='bold', color='#f1f5f9', y=0.995)
plt.tight_layout()
plt.savefig(assets_dir / 'attention_heatmap.png', dpi=300, bbox_inches='tight', facecolor='#1e293b')
plt.close()

# 4. Model Architecture Diagram (Simple)
fig, ax = plt.subplots(figsize=(10, 8), facecolor='#1e293b')
ax.set_facecolor('#0f172a')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Draw architecture components
components = [
    (5, 9, 'Token\nEmbedding', '#6366f1'),
    (5, 7.5, 'Positional\nEmbedding', '#8b5cf6'),
    (5, 6, 'Transformer\nBlock 1-6', '#10b981'),
    (5, 4.5, 'Layer\nNorm', '#f59e0b'),
    (5, 3, 'Output\nProjection', '#ef4444'),
    (5, 1.5, 'Logits', '#6366f1'),
]

for x, y, label, color in components:
    rect = plt.Rectangle((x-1.5, y-0.4), 3, 0.8, facecolor=color, edgecolor='white', linewidth=2, alpha=0.8)
    ax.add_patch(rect)
    ax.text(x, y, label, ha='center', va='center', fontsize=11, fontweight='bold', color='white')

    if y > 1.5:
        ax.arrow(x, y-0.5, 0, -0.4, head_width=0.2, head_length=0.1, fc='white', ec='white', linewidth=2)

ax.text(5, 9.5, 'GPT Model Architecture', ha='center', fontsize=16, fontweight='bold', color='#f1f5f9')

plt.tight_layout()
plt.savefig(assets_dir / 'architecture.png', dpi=300, bbox_inches='tight', facecolor='#1e293b')
plt.close()

print("Generated all demo plots in assets/")
print("  - training_loss.png")
print("  - perplexity.png")
print("  - attention_heatmap.png")
print("  - architecture.png")
