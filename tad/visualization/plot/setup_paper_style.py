import matplotlib.pyplot as plt


def setup_paper_style():
    """
    Journal Standard: Arial/Helvetica.
    Note:
    - Draw.io's default Helvetica is effectively Arial.
    - On Linux systems, Arial may not be available.
    """
    plt.rcParams.update(
        {
            "figure.figsize": [6.8, 4],  # double-column; single-column → [3.5, 2.6]
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
            ],  # Font fallback priority order
            "font.size": 9,
            "axes.labelsize": 10,  # Use the same font size as the main text.
            "axes.titlesize": 10,  # Not used — figure captions are handled by LaTeX \caption{}
            "axes.linewidth": 1.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "grid.linestyle": "--",
            "grid.color": "#bdbdbd",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.3,  # Transparency
            "lines.linewidth": 2.0,
            "lines.markersize": 7,
            "legend.fontsize": 9,
            "legend.frameon": True,
            "legend.framealpha": 0.8,
            "legend.edgecolor": "black",
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        }
    )
