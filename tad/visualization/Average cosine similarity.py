import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Set font configuration (optional, attempting to match the style)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# Data estimation from the chart
layers = [0, 1, 2, 3, 4, 5, 6]

# Series 1: HiFi (Blue line with markers)
hifi_data = [0.85, 0.65, 0.61, 0.55, 0.50, 0.52, 0.48]

# Series 2: ActionFormer (Orange line with markers)
actionformer_data = [0.85, 0.80, 0.88, 0.71, 0.75, 0.74, 0.83]

# Series 3: Raw features (Black constant line)
raw_features_value = 0.87

# Create the plot
plt.figure(figsize=(8, 5))

# Plot HiFi
plt.plot(layers, hifi_data, marker='o', linestyle='-', linewidth=2.5, markersize=8, label='HiFi', color='#1f77b4')

# Plot ActionFormer
plt.plot(layers, actionformer_data, marker='o', linestyle='-', linewidth=2.5, markersize=8, label='ActionFormer', color='#ff7f0e')

# Plot Raw features
plt.hlines(y=raw_features_value, xmin=0, xmax=6, color='black', linewidth=1.5, label='Raw features')

# Configure Grid
plt.grid(True, linestyle='--', linewidth=1, alpha=0.6, color='#bdbdbd')

# Configure Axes limits
plt.ylim(0.3, 1.0)
plt.xlim(-0.2, 6.2)

# Configure Labels
plt.title('Average cosine similarity', fontsize=16, fontweight='bold', pad=10)
plt.xlabel('Layers', fontsize=16, fontweight='bold')
plt.ylabel('Cosine similarity', fontsize=16, fontweight='bold')

# Configure Ticks
plt.xticks(layers, fontsize=12)
plt.yticks(np.arange(0.3, 1.01, 0.1), fontsize=12)

# Configure Legend
plt.legend(loc='lower left', fontsize=12, frameon=True, edgecolor='#bfbfbf')

# Make the layout tight
plt.tight_layout()

# Save the plot
output_path = 'average_cosine_similarity.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_path}")

# Show the plot
plt.show()
