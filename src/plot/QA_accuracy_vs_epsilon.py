import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving files directly
import matplotlib.pyplot as plt
import numpy as np
import os
os.makedirs('results/analysis', exist_ok=True)

# Set publication-quality defaults matching first figure
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

# --- Data from your screenshot ---
epsilon = np.array([0.5, 1, 2, 5, 8, 10])
avg_performance = np.array([0.6325, 0.6325, 0.65, 0.665, 0.66, 0.7125])
std_dev = np.array([0.005, 0.005, 0.00816497, 0.0057735, 0, 0.005])

# --- Plotting ---
# Create a figure and an axes object
fig, ax = plt.subplots(figsize=(9, 5.5))  # Changed to match first figure

# Plot the average performance line with updated styling
ax.plot(epsilon, avg_performance,
        marker='o',
        linewidth=2.5,
        markersize=7,
        label='Average Accuracy',
        color='#06A77D')  # Using color from first figure

# Create the shaded area for the standard deviation
ax.fill_between(
    epsilon,
    avg_performance - std_dev,
    avg_performance + std_dev,
    color='#06A77D',
    alpha=0.2
)

# --- Customization for Academic Paper ---
ax.set_xlabel('$\epsilon_p$', fontsize=22, fontweight='bold')
ax.set_ylabel('QA Accuracy', fontsize=22, fontweight='bold')
ax.set_title('QA Accuracy vs. $\epsilon_p$ (Privacy Budget)',
             fontsize=24, fontweight='bold', pad=15)

# Add a legend
ax.legend(loc='upper left', framealpha=0.95, edgecolor='black', fontsize=22)

# Configure grid and ticks
ax.grid(True, linestyle='--', alpha=0.3, axis='both')
ax.tick_params(axis='both', which='major', labelsize=18)

# Set Y-axis limits to focus the view on the data range
ax.set_ylim(0.6, 0.75)

# Ensure the layout is tight
plt.tight_layout()

# --- Save the figure in high quality ---
plt.savefig('results/analysis/qa_performance_vs_epsilon_styled.pdf', dpi=300, bbox_inches='tight')
plt.savefig('results/analysis/qa_performance_vs_epsilon_styled.png', dpi=300, bbox_inches='tight')

print("Plot saved as 'results/analysis/qa_performance_vs_epsilon_styled.pdf' and '.png'")