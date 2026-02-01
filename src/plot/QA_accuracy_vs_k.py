#!/usr/bin/env python3
"""
Plot QA Performance vs. Number of Retrieved Neighbors (k)
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 16,
    'font.family': 'sans-serif',
    'axes.labelsize': 20,
    'axes.titlesize': 22,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 15,
    'figure.figsize': (9, 5.5),
    'lines.linewidth': 2.5,
    'lines.markersize': 7,
    'grid.alpha': 0.3,
    'axes.grid': True,
})

# Create output directory
os.makedirs('results/analysis', exist_ok=True)

# Data
k_values = [1, 3, 5, 7]
private_accuracy = [0.78, 0.78, 0.78, 0.78]
public_accuracy = [0.89, 0.89, 0.89, 0.89]

# Create figure
fig, ax = plt.subplots(figsize=(9, 5.5))

# Plot lines with markers
ax.plot(k_values, private_accuracy, 'o-', label='Private Data', 
        color='#E74C3C', markersize=7, linewidth=2.5)
ax.plot(k_values, public_accuracy, 's-', label='Public Data', 
        color='#3498DB', markersize=7, linewidth=2.5)

# Set labels and title
ax.set_xlabel('$k$', fontsize=20, fontweight='bold')
ax.set_ylabel('QA Accuracy', fontsize=20, fontweight='bold')
ax.set_title('QA Accuracy vs. $k$ (Number of Retrieved Neighbors)', 
             fontsize=22, fontweight='bold', pad=15)

# Set tick parameters
ax.set_xticks(k_values)
ax.tick_params(axis='both', which='major', labelsize=16)

# Set y-axis limits
ax.set_ylim(0.70, 0.95)

# Grid
ax.grid(True, linestyle='--', alpha=0.3, axis='both')

# Legend
ax.legend(loc='best', framealpha=0.95, edgecolor='black', fontsize=22)

# Tight layout
plt.tight_layout()

# Save figure
plt.savefig('results/analysis/qa_vs_k_neighbors.png', dpi=300, bbox_inches='tight')
plt.savefig('results/analysis/qa_vs_k_neighbors.pdf', bbox_inches='tight')
print("✅ Saved: QA Performance vs. k")
plt.close()

print("\n" + "="*80)
print("📊 QA PERFORMANCE vs. k ANALYSIS")
print("="*80)
print(f"\nPrivate Data Accuracy: {private_accuracy[0]:.2f} (constant across all k)")
print(f"Public Data Accuracy: {public_accuracy[0]:.2f} (constant across all k)")
print(f"Performance Gap: {public_accuracy[0] - private_accuracy[0]:.2f}")
print("\n✅ Figure saved to results/analysis/")
print("="*80)