import matplotlib.pyplot as plt


def get_figsize_from_pt(width_pt, ratio=1.618, fraction=1.0):
    r"""
    Convert LaTeX width (pt) to Matplotlib figsize (inches).

    Parameters:
        width_pt (float): LaTeX \columnwidth or \textwidth in points (pt),
                          use \typeout{COLUMNWIDTH = \the\columnwidth} or \typeout{TEXTWIDTH = \the\textwidth}
        ratio (float): Aspect ratio (width/height), default golden ratio 1.618
        fraction (float): Fraction of width to use (e.g., 0.95 for margins)

    Returns:
        (width_inch, height_inch): Matplotlib figure dimensions in inches

    Note:
        1 inch = 25.4 mm
        = 72.27 pt   (TeX pt)
        = 72 bp      (PostScript / PDF / Word /matplotlib / MATLAB pt)

        -------------------------------------------------------------------
        Original comment from matplotlib.backends.backend_pdf:

        It's better to use only one unit for all coordinates, since the
        arithmetic in latex seems to produce inaccurate conversions.

        latex_pt_to_in = 1. / 72.27
        latex_in_to_pt = 1. / latex_pt_to_in
        mpl_pt_to_in = 1. / 72.
        mpl_in_to_pt = 1. / mpl_pt_to_in
        -------------------------------------------------------------------

        A4 (210 * 297 mm)
        = 8.27 * 11.69 inch
        ≈ 598 * 845 pt (TeX pt)

        Letter (216 * 279 mm)
        = 8.5 * 11 inch
        ≈ 615 * 795 pt (TeX pt)

    """
    inches_per_pt = 1 / 72.27  # latex_pt_to_in
    width_inch = width_pt * inches_per_pt * fraction
    height_inch = width_inch / ratio
    return (width_inch, height_inch)


def setup_paper_style(textwidth, fraction=0.98):  # 0.98 \textwidth

    figsize = get_figsize_from_pt(
        textwidth, ratio=1.618, fraction=fraction
    )  # double-column; single-column → textwidth / 2

    plt.rcParams.update(
        {
            "figure.figsize": figsize,
            # Journal Standard: Arial/Helvetica.
            # Note:
            # - Draw.io's default Helvetica is effectively Arial.
            # - On Linux systems, Arial may not be available.
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
            ],  # Font fallback priority order
            "font.size": 9,
            "axes.labelsize": 10,  # Use the same font size as the main text.
            "axes.titlesize": 10,  # Not used - figure captions are handled by LaTeX \caption{}
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
            "savefig.pad_inches": 0,
        }
    )
