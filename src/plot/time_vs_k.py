import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory
os.makedirs('results/analysis', exist_ok=True)

# Data extracted from the statistics
k_values = [1, 3, 5, 7, 9, 10, 15, 20, 30, 50]

# Total time (avg and std)
total_time_avg = [0.11841986617631141, 0.11893770859999218, 0.12074971131130553, 
                  0.11990344769395599, 0.11956788007209934, 0.12047738148501784,
                  0.13039464987642796, 0.11899189584914506, 0.12006144565763074,
                  0.11816878294381951]
total_time_std = [0.0009181355193798265, 0.0009335163756374182, 0.0009715448056128675,
                  0.0009173621992961075, 0.0009604739681594632, 0.0009125594929906793,
                  0.0030034479594115196, 0.0010616844866502113, 0.001054759432914508,
                  0.0009765763184421648]

# FAISS time (avg and std)
faiss_time_avg = [0.04592176539174461, 0.04603082374157031, 0.04735557753817055,
                  0.047680179690125765, 0.047015646067036204, 0.047544531410558026,
                  0.0527797897364439, 0.045720222321841995, 0.04751992863738777,
                  0.04552505841367534]
faiss_time_std = [0.0004715021743648208, 0.0005420044646051724, 0.0005470033574113345,
                  0.0005408574860405608, 0.0005502218137572186, 0.0005407766801466552,
                  0.0018281056092217425, 0.0005587250326699128, 0.000525089289564678,
                  0.0004907228516081764]

# Distance computation time (avg and std)
dist_time_avg = [0.0047467327966046, 0.004758823365944891, 0.0046781566359203434,
                 0.004625436636923458, 0.004723892706047202, 0.004635680771242189,
                 0.005742878732790324, 0.004759649094756732, 0.004654649717725542,
                 0.004774010308259804]
dist_time_std = [0.00011517229580647025, 0.00011939397520760708, 0.0001157246590869168,
                 4.860704709842557e-05, 0.00011960249274388267, 4.196974585739906e-05,
                 0.00040892693883895286, 0.00012967703618493016, 6.088191934004268e-05,
                 0.000132721420817773]

# Convert to numpy arrays
k_values = np.array(k_values)
total_time_avg = np.array(total_time_avg) * 1000  # Convert to milliseconds
total_time_std = np.array(total_time_std) * 1000
faiss_time_avg = np.array(faiss_time_avg) * 1000
faiss_time_std = np.array(faiss_time_std) * 1000
dist_time_avg = np.array(dist_time_avg) * 1000
dist_time_std = np.array(dist_time_std) * 1000

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot all three curves at ACTUAL k values (not evenly spaced)
ax.plot(k_values, total_time_avg, 'o-', linewidth=2, markersize=6, label='Total Retrieval Time', color='#2E86AB')
ax.fill_between(k_values, 
                total_time_avg - total_time_std, 
                total_time_avg + total_time_std, 
                alpha=0.2, color='#2E86AB')

ax.plot(k_values, faiss_time_avg, 's-', linewidth=2, markersize=6, label='FAISS Time', color='#A23B72')
ax.fill_between(k_values, 
                faiss_time_avg - faiss_time_std, 
                faiss_time_avg + faiss_time_std, 
                alpha=0.2, color='#A23B72')

ax.plot(k_values, dist_time_avg, '^-', linewidth=2, markersize=6, label='kNN Prob Distribution Computation Time', color='#F18F01')
ax.fill_between(k_values, 
                dist_time_avg - dist_time_std, 
                dist_time_avg + dist_time_std, 
                alpha=0.2, color='#F18F01')

ax.set_xlabel('$k$', fontsize=20, fontweight='bold')
ax.set_ylabel('Time (ms)', fontsize=20, fontweight='bold')
ax.set_title('kNN Retrieval Time vs $k$', fontsize=20, fontweight='bold')


# Set EVENLY SPACED x-ticks for the grid (e.g., every 10 units)
even_xticks = np.arange(0, 60, 10)  # 0, 10, 20, 30, 40, 50
ax.set_xticks(even_xticks)
ax.set_xlim(-2, 55)

# Get the y-axis limits
y_min, y_max = ax.get_ylim()

# Add projection lines from data points to x-axis (all the way to bottom)
for i, k_val in enumerate(k_values):
    # Projection line from point all the way down to the bottom
    ax.plot([k_val, k_val], [y_min, total_time_avg[i]], 
            linestyle=':', color='gray', alpha=0.4, linewidth=1)

# Now add labels at the very bottom
for k_val in k_values:
    ax.text(k_val, y_min - 3, f'{k_val}',
            ha='center', va='top', fontsize=10, color='darkblue',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                     edgecolor='lightgray', alpha=0.8, linewidth=0.5))

ax.legend(loc='best', fontsize=16, markerscale=1.5)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('results/analysis/timing_analysis_cost_of_k.png', dpi=300, bbox_inches='tight')
plt.show()