import matplotlib.pyplot as plt

def setup_paper_style():
    plt.rcParams.update({
        # Journal Standard: Arial/Helvetica. 
        # Note: draw.io's default Helvetica is effectively Arial.
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], # Font fallback priority order
        'font.size': 11,
        
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,

        'xtick.labelsize': 11,
        'ytick.labelsize': 11,

        'grid.linewidth': 0.8,     
        'grid.alpha': 0.3,         # Grid line transparency (0 = fully transparent, 1 = opaque)

        'lines.linewidth': 3.0,
        'lines.markersize': 10, 

        'legend.fontsize': 11,
        'legend.frameon': True,
        'legend.framealpha': 0.8, 
        'legend.edgecolor': 'black', 

        'savefig.dpi': 600,
        'savefig.bbox': 'tight',   
        'savefig.pad_inches': 0.1, 
    })
