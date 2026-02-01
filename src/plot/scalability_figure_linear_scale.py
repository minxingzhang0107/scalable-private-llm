#!/usr/bin/env python3
"""
Plot KNN-LM Performance Analysis for Academic Paper
Updated with complete data from screenshot and publication-quality formatting.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Set publication-quality defaults for DOUBLE-COLUMN layout
plt.rcParams.update({
    'font.size': 16,           # Increased from 12
    'font.family': 'sans-serif',
    'axes.labelsize': 20,      # Increased from 14
    'axes.titlesize': 22,      # Increased from 15
    'xtick.labelsize': 16,     # Increased from 11
    'ytick.labelsize': 16,     # Increased from 11
    'legend.fontsize': 15,     # Increased from 11
    'figure.figsize': (9, 5.5),
    'lines.linewidth': 2.5,
    'lines.markersize': 7,
    'grid.alpha': 0.3,
    'axes.grid': True,
})

# Create output directory
os.makedirs('results/analysis', exist_ok=True)

# ============================================================================
# DATA FROM SCREENSHOT
# ============================================================================

# Training data sizes
training_sizes = np.array([1, 5, 10, 50, 100, 280, 460, 640, 820, 1000])  # in thousands

# Number of pairs (from screenshot)
num_pairs = np.array([19577, 98025, 195749, 979283, 1959096, 5485158, 9012146, 12538342, 16065725, 19591072])
num_pairs_millions = num_pairs / 1e6  # Convert to millions

# Datastore sizes in bytes (from screenshot)
datastore_bytes = np.array([308.29e6, 1.50e9, 3.00e9, 14.97e9, 29.94e9, 83.79e9, 137.66e9, 191.51e9, 245.38e9, 299.22e9])
datastore_gb = datastore_bytes / 1e9  # Convert to GB

# Datastore Building Times (in minutes)
total_time = np.array([3.24, 16.36, 31.41, 162.89, 320.65, 909.03, 1468.4, 2069.61, 2690.21, 3411.73])
embedding_time = np.array([3.14, 15.85, 30.29, 157.51, 308.97, 874.82, 1405.61, 1974.48, 2568.32, 3062.06])
faiss_operation = np.array([0.05, 0.28, 0.61, 2.99, 6.59, 21.19, 40.10, 63.75, 82.84, 300.63])

# Answer Generation Times
per_query_avg = np.array([1.4173, 1.5840, 1.5888, 2.2160, 2.5249, 3.4315, 4.7353, 5.4798, 5.7289, 7.0102])
per_query_std = np.array([0.0671, 0.0718, 0.08, 0.1352, 0.1537, 0.3010, 0.4047, 0.4816, 0.5002, 0.8389])

per_token_avg = np.array([0.0736, 0.0822, 0.0826, 0.1156, 0.1318, 0.1781, 0.2467, 0.2861, 0.2981, 0.3631])
per_token_std = np.array([0.0028, 0.0036, 0.0042, 0.0096, 0.0113, 0.0273, 0.0435, 0.0560, 0.0598, 0.0801])

faiss_search_avg = np.array([0.0049, 0.0126, 0.0187, 0.0451, 0.0667, 0.1083, 0.1820, 0.2194, 0.2279, 0.2998])
faiss_search_std = np.array([0.0008, 0.0021, 0.0031, 0.0092, 0.0113, 0.0271, 0.0429, 0.0551, 0.0582, 0.0801])

# ============================================================================
# FIGURE 1: DATASTORE BUILD TIME (linear scale with even grid, uneven points)
# ============================================================================

# fig1, ax1 = plt.subplots(figsize=(9, 5.5))
fig1, ax1 = plt.subplots(figsize=(9, 5.3))

# Plot at ACTUAL x positions (uneven)
ax1.plot(num_pairs_millions, total_time, 'o-', label='Total Build Time', 
         color='#2E86AB', markersize=7, linewidth=2.5)
ax1.plot(num_pairs_millions, embedding_time, 's-', label='Embedding Computation', 
         color='#A23B72', markersize=7, linewidth=2.5)

ax1.set_xlabel('Total Number of Pairs (Millions)', fontsize=22, fontweight='bold')
ax1.set_ylabel('Time (minutes)', fontsize=22, fontweight='bold')
ax1.set_title('Database Build Time vs. Number of Pairs', fontsize=24, fontweight='bold', pad=15)

# Set EVENLY SPACED x-ticks (grid at 0, 5, 10, 15, 20)
even_xticks = np.arange(0, 21, 5)
ax1.set_xticks(even_xticks)
ax1.set_xlim(-0.5, 20.5)

# Increase tick label sizes
ax1.tick_params(axis='both', which='major', labelsize=18)

# Add projection lines and x-value labels for points 4 onwards (skip first 3)
for idx in range(3, len(num_pairs_millions)):
    # Projection line from point to x-axis
    ax1.plot([num_pairs_millions[idx], num_pairs_millions[idx]], 
             [0, embedding_time[idx]], 
             linestyle=':', color='gray', alpha=0.5, linewidth=1)
    # X-value label at bottom
    ax1.text(num_pairs_millions[idx], -150, f'{num_pairs_millions[idx]:.1f}M',
             ha='center', va='top', fontsize=13, color='darkblue', 
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                      edgecolor='lightgray', alpha=0.7, linewidth=0.5))

# Add percentage annotations for embedding computation
embedding_percentages = (embedding_time / total_time) * 100
# Show one percentage for first 3 points (at index 1)
ax1.annotate(f'{embedding_percentages[1]:.1f}%', 
            xy=(num_pairs_millions[1], embedding_time[1]),
            xytext=(0, 8), textcoords='offset points',
            ha='center', fontsize=13, 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='gray', alpha=0.8, linewidth=0.8))

# Show all percentages for points 4 onwards
for idx in range(3, len(num_pairs_millions)):
    ax1.annotate(f'{embedding_percentages[idx]:.1f}%', 
                xy=(num_pairs_millions[idx], embedding_time[idx]),
                xytext=(0, 8), textcoords='offset points',
                ha='center', fontsize=13, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='gray', alpha=0.8, linewidth=0.8))

# Grid and legend
ax1.grid(True, linestyle='--', alpha=0.3, axis='both')
ax1.legend(loc='upper left', framealpha=0.95, edgecolor='black', fontsize=22)

plt.tight_layout()
plt.savefig('results/analysis/fig1_datastore_build_time.png', dpi=300, bbox_inches='tight')
plt.savefig('results/analysis/fig1_datastore_build_time.pdf', bbox_inches='tight')
print("✅ Saved: Figure 1 - Datastore Build Time")
plt.close()

# ============================================================================
# FIGURE 2: PER-QUERY GENERATION TIME (linear scale with even grid, uneven points)
# ============================================================================

fig2, ax2 = plt.subplots(figsize=(9, 5.5))

# Plot at ACTUAL x positions (uneven)
ax2.errorbar(num_pairs_millions, per_query_avg, yerr=per_query_std, 
             fmt='o-', label='Per-Query Generation Time', 
             color='#F18F01', markersize=7, 
             linewidth=2.5, capsize=4, capthick=2)

ax2.set_xlabel('Total Number of Pairs (Millions)', fontsize=22, fontweight='bold')
ax2.set_ylabel('Time (seconds)', fontsize=22, fontweight='bold')
ax2.set_title('Per-Query Generation Time vs. Number of Pairs', 
              fontsize=24, fontweight='bold', pad=15)

# Set EVENLY SPACED x-ticks (grid at 0, 5, 10, 15, 20)
even_xticks = np.arange(0, 21, 5)
ax2.set_xticks(even_xticks)
ax2.set_xlim(-0.5, 20.5)

# Increase tick label sizes
ax2.tick_params(axis='both', which='major', labelsize=18)

# Add projection lines and x-value labels for points 4 onwards (skip first 3)
for idx in range(3, len(num_pairs_millions)):
    # Projection line from point to x-axis
    ax2.plot([num_pairs_millions[idx], num_pairs_millions[idx]], 
             [0, per_query_avg[idx]], 
             linestyle=':', color='gray', alpha=0.5, linewidth=1)
    # X-value label at bottom
    ax2.text(num_pairs_millions[idx], -0.35, f'{num_pairs_millions[idx]:.1f}M',
             ha='center', va='top', fontsize=13, color='darkblue', 
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                      edgecolor='lightgray', alpha=0.7, linewidth=0.5))

# Grid and legend
ax2.grid(True, linestyle='--', alpha=0.3, axis='both')
ax2.legend(loc='upper left', framealpha=0.95, edgecolor='black', fontsize=22)

# Add error shading
ax2.fill_between(num_pairs_millions, 
                 per_query_avg - per_query_std, 
                 per_query_avg + per_query_std, 
                 alpha=0.2, color='#F18F01')

plt.tight_layout()
plt.savefig('results/analysis/fig2_per_query_generation_time.png', dpi=300, bbox_inches='tight')
plt.savefig('results/analysis/fig2_per_query_generation_time.pdf', bbox_inches='tight')
print("✅ Saved: Figure 2 - Per-Query Generation Time")
plt.close()

# ============================================================================
# FIGURE 3: PER-TOKEN GENERATION TIME WITH FAISS BREAKDOWN (linear scale)
# ============================================================================

fig3, ax3 = plt.subplots(figsize=(9, 5.5))

# Plot at ACTUAL x positions (uneven)
ax3.errorbar(num_pairs_millions, per_token_avg, yerr=per_token_std, 
             fmt='o-', label='Total Per-Token Time', 
             color='#06A77D', markersize=7, 
             linewidth=2.5, capsize=4, capthick=2, zorder=3)

ax3.fill_between(num_pairs_millions, 
                 per_token_avg - per_token_std, 
                 per_token_avg + per_token_std, 
                 alpha=0.2, color='#06A77D', zorder=1)

# Plot FAISS search time
ax3.errorbar(num_pairs_millions, faiss_search_avg, yerr=faiss_search_std, 
             fmt='s-', label='FAISS Search Time', 
             color='#D62246', markersize=7, 
             linewidth=2.5, capsize=4, capthick=2, zorder=3)

ax3.fill_between(num_pairs_millions, 
                 faiss_search_avg - faiss_search_std, 
                 faiss_search_avg + faiss_search_std, 
                 alpha=0.2, color='#D62246', zorder=1)

ax3.set_xlabel('Total Number of Pairs (Millions)', fontsize=22, fontweight='bold')
ax3.set_ylabel('Time (seconds)', fontsize=22, fontweight='bold')
ax3.set_title('Per-Token Generation Time vs. Number of Pairs', 
              fontsize=24, fontweight='bold', pad=15)

# Set EVENLY SPACED x-ticks (grid at 0, 5, 10, 15, 20)
even_xticks = np.arange(0, 21, 5)
ax3.set_xticks(even_xticks)
ax3.set_xlim(-0.5, 20.5)

# Increase tick label sizes
ax3.tick_params(axis='both', which='major', labelsize=18)

# Add projection lines and x-value labels for points 4 onwards (skip first 3)
for idx in range(3, len(num_pairs_millions)):
    # Projection line from point to x-axis
    ax3.plot([num_pairs_millions[idx], num_pairs_millions[idx]], 
             [0, per_token_avg[idx]], 
             linestyle=':', color='gray', alpha=0.5, linewidth=1)
    # X-value label at bottom
    ax3.text(num_pairs_millions[idx], -0.02, f'{num_pairs_millions[idx]:.1f}M',
             ha='center', va='top', fontsize=13, color='darkblue', 
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                      edgecolor='lightgray', alpha=0.7, linewidth=0.5))

# Add FAISS percentage annotations
faiss_percentages = (faiss_search_avg / per_token_avg) * 100
# Show one percentage for first 3 points (at index 1)
ax3.annotate(f'{faiss_percentages[1]:.1f}%', 
            xy=(num_pairs_millions[1], faiss_search_avg[1]),
            xytext=(0, 8), textcoords='offset points',
            ha='center', fontsize=13, 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='gray', alpha=0.8, linewidth=0.8))

# Show all percentages for points 4 onwards
for idx in range(3, len(num_pairs_millions)):
    ax3.annotate(f'{faiss_percentages[idx]:.1f}%', 
                xy=(num_pairs_millions[idx], faiss_search_avg[idx]),
                xytext=(0, 8), textcoords='offset points',
                ha='center', fontsize=13, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='gray', alpha=0.8, linewidth=0.8))

# Grid and legend
ax3.grid(True, linestyle='--', alpha=0.3, axis='both')
ax3.legend(loc='upper left', framealpha=0.95, edgecolor='black', fontsize=20)

plt.tight_layout()
plt.savefig('results/analysis/fig3_per_token_generation_time.png', dpi=300, bbox_inches='tight')
plt.savefig('results/analysis/fig3_per_token_generation_time.pdf', bbox_inches='tight')
print("✅ Saved: Figure 3 - Per-Token Generation Time")
plt.close()

# ============================================================================
# FIGURE 3b: NON-FAISS COMPUTATION TIME (Total - FAISS) (linear scale)
# ============================================================================

fig3b, ax3b = plt.subplots(figsize=(9, 5.5))

# Calculate non-FAISS time
non_faiss_time = per_token_avg - faiss_search_avg
non_faiss_std = np.sqrt(per_token_std**2 + faiss_search_std**2)  # Error propagation

# Plot at ACTUAL x positions (uneven)
ax3b.errorbar(num_pairs_millions, non_faiss_time, yerr=non_faiss_std, 
             fmt='o-', label='Non-FAISS Computation', 
             color='#9B59B6', markersize=7, 
             linewidth=2.5, capsize=4, capthick=2)

ax3b.fill_between(num_pairs_millions, 
                  non_faiss_time - non_faiss_std, 
                  non_faiss_time + non_faiss_std, 
                  alpha=0.2, color='#9B59B6')

ax3b.set_xlabel('Total Number of Pairs (Millions)', fontsize=22, fontweight='bold')
ax3b.set_ylabel('Time (seconds)', fontsize=22, fontweight='bold')
ax3b.set_title('Non-FAISS Computation Time vs. Number of Pairs', 
              fontsize=24, fontweight='bold', pad=15)

# Set EVENLY SPACED x-ticks (grid at 0, 5, 10, 15, 20)
even_xticks = np.arange(0, 21, 5)
ax3b.set_xticks(even_xticks)
ax3b.set_xlim(-0.5, 20.5)

# Increase tick label sizes
ax3b.tick_params(axis='both', which='major', labelsize=18)

# Get y-axis limits to calculate proper bottom position
y_min, y_max = ax3b.get_ylim()

# Add projection lines and x-value labels for points 4 onwards (skip first 3)
for idx in range(3, len(num_pairs_millions)):
    # Projection line from point ALL THE WAY DOWN to x-axis (y=y_min)
    ax3b.plot([num_pairs_millions[idx], num_pairs_millions[idx]], 
             [y_min, non_faiss_time[idx]], 
             linestyle=':', color='gray', alpha=0.5, linewidth=1)
    # X-value label at bottom (below y_min)
    ax3b.text(num_pairs_millions[idx], y_min - (y_max - y_min) * 0.04, 
             f'{num_pairs_millions[idx]:.1f}M',
             ha='center', va='top', fontsize=13, color='darkblue', 
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                      edgecolor='lightgray', alpha=0.7, linewidth=0.5))

# Grid and legend
ax3b.grid(True, linestyle='--', alpha=0.3, axis='both')
ax3b.legend(loc='upper left', framealpha=0.95, edgecolor='black', fontsize=15)

plt.tight_layout()
plt.savefig('results/analysis/fig3b_non_faiss_computation_time.png', dpi=300, bbox_inches='tight')
plt.savefig('results/analysis/fig3b_non_faiss_computation_time.pdf', bbox_inches='tight')
print("✅ Saved: Figure 3b - Non-FAISS Computation Time")
plt.close()

# ============================================================================
# FIGURE 4: TRAINING DATA SIZE vs. NUMBER OF PAIRS
# ============================================================================

fig4, ax4 = plt.subplots(figsize=(9, 5.5))

# Plot training size vs number of pairs
ax4.plot(training_sizes, num_pairs_millions, 'o-', 
         color='#E74C3C', markersize=7, linewidth=2.5)

ax4.set_xlabel('Number of Private Facts (Thousands)', fontsize=16, fontweight='bold')
ax4.set_ylabel('Total Number of Pairs (Millions)', fontsize=16, fontweight='bold')
ax4.set_title('Number of Pairs vs. Number of Private Facts', 
              fontsize=17, fontweight='bold', pad=15)

# Increase tick label sizes
ax4.tick_params(axis='both', which='major', labelsize=18)

# Grid
ax4.grid(True, linestyle='--', alpha=0.3, axis='both')

# Add projection lines and labels for points 4 onwards (skip first 3)
y_min, y_max = ax4.get_ylim()
for idx in range(3, len(training_sizes)):
    # Projection line from point down to x-axis
    ax4.plot([training_sizes[idx], training_sizes[idx]], 
             [y_min, num_pairs_millions[idx]], 
             linestyle=':', color='gray', alpha=0.5, linewidth=1)
    # X-value label at bottom
    ax4.text(training_sizes[idx], y_min - (y_max - y_min) * 0.04, 
             f'{training_sizes[idx]}K',
             ha='center', va='top', fontsize=13, color='darkblue', 
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                      edgecolor='lightgray', alpha=0.7, linewidth=0.5))

plt.tight_layout()
plt.savefig('results/analysis/fig4_pairs_vs_training_size.png', dpi=300, bbox_inches='tight')
plt.savefig('results/analysis/fig4_pairs_vs_training_size.pdf', bbox_inches='tight')
print("✅ Saved: Figure 4 - Pairs vs. Training Size")
plt.close()

# ============================================================================
# FIGURE 5: TRAINING DATA SIZE vs. DATASTORE SIZE (GB)
# ============================================================================

fig5, ax5 = plt.subplots(figsize=(9, 5.5))

# Plot training size vs datastore size
ax5.plot(training_sizes, datastore_gb, 'o-', 
         color='#3498DB', markersize=7, linewidth=2.5)

ax5.set_xlabel('Number of Private Facts (Thousands)', fontsize=16, fontweight='bold')
ax5.set_ylabel('Database Size (GB)', fontsize=16, fontweight='bold')
ax5.set_title('Database Size vs. Number of Private Facts', 
              fontsize=17, fontweight='bold', pad=15)

# Increase tick label sizes
ax5.tick_params(axis='both', which='major', labelsize=18)

# Grid
ax5.grid(True, linestyle='--', alpha=0.3, axis='both')

# Add projection lines and labels for points 4 onwards (skip first 3)
y_min, y_max = ax5.get_ylim()
for idx in range(3, len(training_sizes)):
    # Projection line from point down to x-axis
    ax5.plot([training_sizes[idx], training_sizes[idx]], 
             [y_min, datastore_gb[idx]], 
             linestyle=':', color='gray', alpha=0.5, linewidth=1)
    # X-value label at bottom
    ax5.text(training_sizes[idx], y_min - (y_max - y_min) * 0.04, 
             f'{training_sizes[idx]}K',
             ha='center', va='top', fontsize=13, color='darkblue', 
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                      edgecolor='lightgray', alpha=0.7, linewidth=0.5))

plt.tight_layout()
plt.savefig('results/analysis/fig5_datastore_size_vs_training_size.png', dpi=300, bbox_inches='tight')
plt.savefig('results/analysis/fig5_datastore_size_vs_training_size.pdf', bbox_inches='tight')
print("✅ Saved: Figure 5 - Datastore Size vs. Training Size")
plt.close()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*80)
print("📊 SUMMARY STATISTICS")
print("="*80)

print("\n🔨 Datastore Build Time Analysis:")
print(f"   Embedding % of total (avg): {np.mean(embedding_percentages):.1f}%")
print(f"   Embedding % range: {np.min(embedding_percentages):.1f}% - {np.max(embedding_percentages):.1f}%")
print(f"   Build time scaling (1K→1M): {total_time[-1]/total_time[0]:.1f}×")

print("\n⚡ Generation Time Analysis:")
print(f"   Per-query scaling (1K→1M): {per_query_avg[-1]/per_query_avg[0]:.2f}×")
print(f"   Per-token scaling (1K→1M): {per_token_avg[-1]/per_token_avg[0]:.2f}×")
print(f"   FAISS search scaling (1K→1M): {faiss_search_avg[-1]/faiss_search_avg[0]:.1f}×")

print("\n📈 FAISS Overhead (% of per-token time):")
faiss_percentages = (faiss_search_avg / per_token_avg) * 100
for i, size in enumerate(training_sizes):
    print(f"   {size}K: {faiss_percentages[i]:.1f}%")

print("\n📏 Scaling Relationships:")
print(f"   Pairs per training example: {np.mean(num_pairs / (training_sizes * 1000)):.1f} avg")
print(f"   Datastore size per 1K training: {np.mean(datastore_gb / training_sizes):.2f} GB avg")

print("\n✅ All figures saved to results/analysis/")
print("   - Figure 1: Datastore Build Time (with embedding %)")
print("   - Figure 2: Per-Query Generation Time")
print("   - Figure 3: Per-Token Generation Time (total + FAISS)")
print("   - Figure 3b: Non-FAISS Computation Time")
print("   - Figure 4: Pairs vs. Training Size")
print("   - Figure 5: Datastore Size vs. Training Size")
print("="*80)

# ============================================================================
# FIGURE 6: COMBINED PER-TOKEN AND NON-FAISS COMPUTATION TIME (2 subplots)
# ============================================================================

# ============================================================================
# FIGURE 6: COMBINED PER-TOKEN, FAISS, AND NON-FAISS ON SAME PLOT
# ============================================================================

fig6, ax6 = plt.subplots(figsize=(9, 5.5))

# Calculate non-FAISS time
non_faiss_time = per_token_avg - faiss_search_avg
non_faiss_std = np.sqrt(per_token_std**2 + faiss_search_std**2)  # Error propagation

# Plot Total Per-Token Time
ax6.errorbar(num_pairs_millions, per_token_avg, yerr=per_token_std, 
             fmt='o-', label='Total Per-Token Time', 
             color='#06A77D', markersize=7, 
             linewidth=2.5, capsize=4, capthick=2, zorder=3)

ax6.fill_between(num_pairs_millions, 
                 per_token_avg - per_token_std, 
                 per_token_avg + per_token_std, 
                 alpha=0.2, color='#06A77D', zorder=1)

# Plot FAISS Search Time
ax6.errorbar(num_pairs_millions, faiss_search_avg, yerr=faiss_search_std, 
             fmt='s-', label='FAISS Search Time', 
             color='#D62246', markersize=7, 
             linewidth=2.5, capsize=4, capthick=2, zorder=3)

ax6.fill_between(num_pairs_millions, 
                 faiss_search_avg - faiss_search_std, 
                 faiss_search_avg + faiss_search_std, 
                 alpha=0.2, color='#D62246', zorder=1)

# Plot Non-FAISS Computation Time
ax6.errorbar(num_pairs_millions, non_faiss_time, yerr=non_faiss_std, 
             fmt='^-', label='Non-FAISS Computation', 
             color='#9B59B6', markersize=7, 
             linewidth=2.5, capsize=4, capthick=2, zorder=3)

ax6.fill_between(num_pairs_millions, 
                 non_faiss_time - non_faiss_std, 
                 non_faiss_time + non_faiss_std, 
                 alpha=0.2, color='#9B59B6', zorder=1)

ax6.set_xlabel('Total Number of Pairs (Millions)', fontsize=22, fontweight='bold')
ax6.set_ylabel('Time (seconds)', fontsize=22, fontweight='bold')
ax6.set_title('Per-Token Time Breakdown vs. Number of Pairs', 
              fontsize=24, fontweight='bold', pad=15)

# Set EVENLY SPACED x-ticks (grid at 0, 5, 10, 15, 20)
even_xticks = np.arange(0, 21, 5)
ax6.set_xticks(even_xticks)
ax6.set_xlim(-0.5, 20.5)

# Increase tick label sizes
ax6.tick_params(axis='both', which='major', labelsize=18)

# Add projection lines and x-value labels for points 4 onwards (skip first 3)
for idx in range(3, len(num_pairs_millions)):
    # Projection line from point to x-axis
    ax6.plot([num_pairs_millions[idx], num_pairs_millions[idx]], 
             [0, per_token_avg[idx]], 
             linestyle=':', color='gray', alpha=0.5, linewidth=1)
    # X-value label at bottom
    ax6.text(num_pairs_millions[idx], -0.02, f'{num_pairs_millions[idx]:.1f}M',
             ha='center', va='top', fontsize=13, color='darkblue', 
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                      edgecolor='lightgray', alpha=0.7, linewidth=0.5))

# Grid and legend
ax6.grid(True, linestyle='--', alpha=0.3, axis='both')
ax6.legend(loc='upper left', framealpha=0.95, edgecolor='black', fontsize=17)

plt.tight_layout()
plt.savefig('results/analysis/fig6_combined_per_token_breakdown.png', dpi=300, bbox_inches='tight')
plt.savefig('results/analysis/fig6_combined_per_token_breakdown.pdf', bbox_inches='tight')
print("✅ Saved: Figure 6 - Combined Per-Token Time Breakdown")
plt.close()