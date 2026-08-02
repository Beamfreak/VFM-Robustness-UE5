import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
from pathlib import Path

# Data Definition based on Table \ref{tab:scaling_jumps}
# Format: Model Family: [ (Scale_Name, Params_Millions, Normal_Acc, Plastic_Acc), ... ]
# Note: Params are approximate based on standard model sizes for visualization scaling on X-axis

data = {
    "DINOv1": [
        ("Small", 22, 7.3, 7.0),
        ("Base", 86, 8.7, 8.8)
    ],
    "DINOv2": [
        ("Small", 22, 24.7, 20.4),
        ("Base", 86, 31.7, 26.4),
        ("Large", 304, 38.6, 30.9)
    ],
    "DINOv3": [
        ("Small", 22, 15.5, 15.0),
        ("Base", 86, 26.8, 8.2),  
        ("Large", 303, 38.3, 36.6) 
    ],
    "CLIP": [
        ("Base", 149, 25.7, 23.4),
        ("Large", 420, 34.4, 32.3)
    ],
    "Hiera": [
        ("Small", 35, 17.6, 18.5),
        ("Base", 52, 20.1, 20.1),
        ("Large", 214, 25.0, 24.2)
    ],
    "Swin": [
        ("Small", 50, 19.5, 20.4),
        ("Base", 88, 21.4, 20.0),
        ("Large", 197, 23.2, 20.5)
    ],
    "ViT": [
        ("Base", 87, 13.2, 13.6),
        ("Large", 304, 15.7, 16.9)
    ]
}

# Plot Configuration
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

# Color Map for consistency
colors = {
    "DINOv1": "#8c564b", # Brown
    "DINOv2": "#66b3ff", # Lighter blue
    "DINOv3": "#1f5fbf", # Darker Blue
    "CLIP": "#2ca02c", # Green
    "Hiera": "#ff7f0e", # Orange
    "Swin": "#d62728", # Red
    "ViT": "#9467bd" # Purple
}

linestyle_self_sup = "-"
linestyle_sup = "--"

titles = ["Multi-Factor Dataset", "Multi-Color Dataset"]
y_labels = ["Top-1 Accuracy (%)", "Top-1 Accuracy (%)"]

for ax_idx, dataset_col in enumerate([2, 3]): # 2=Normal, 3=Plastic
    ax = axs[ax_idx]
    ax.set_title(titles[ax_idx], fontsize=12, fontweight='bold')
    ax.set_xlabel("Parameter Scale (Millions)", fontsize=11)
    if ax_idx == 0:
        ax.set_ylabel("Top-1 Accuracy (%)", fontsize=11)
    
    ax.set_xscale('log')
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.set_axisbelow(True)
    
    # Set Y limits for better visualization
    ax.set_ylim(0, 45)
    ax.set_yticks(np.arange(0, 45, 5))

    # Set exactly three x-ticks (log-spaced): min, mid, max for clarity
    all_params = sorted({p[1] for points in data.values() for p in points})
    if len(all_params) > 0:
        min_p, max_p = all_params[0], all_params[-1]
        ticks = np.geomspace(min_p, max_p, num=3)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(int(round(t))) for t in ticks])
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

    for family, points in data.items():
        print(f"Plotting {family} for dataset column {dataset_col}")
        params = [p[1] for p in points]
        accs = [p[dataset_col] for p in points]
        labels = [p[0] for p in points]
        
        color = colors[family]
        ls = "--"
        linewidth = 1.5
        marker = 'o'
        
        
        # Plot line
        ax.plot(params, accs, marker=marker, linestyle=ls, linewidth=linewidth, 
                color=color, label=family, markersize=6)
        
        
# Create a unified legend (include all plotted families)
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05),
           ncol=3, frameon=False, fontsize=10)

plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Make room for legend
base_dir = Path(__file__).resolve().parent.parent
out_dir = base_dir / 'results' / 'plots'
out_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(out_dir / "5.6_scaling_curves.pdf", bbox_inches='tight')
plt.savefig(out_dir / "5.6_scaling_curves.png", bbox_inches='tight', dpi=300)
plt.show()