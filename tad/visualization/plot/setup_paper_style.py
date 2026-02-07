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


def setup_paper_style(textwidth, ratio, fraction, font_size_tex=10, font_size_main=9, line_width_axis=0.5):
    figsize = get_figsize_from_pt(textwidth, ratio, fraction)  # double-column; single-column → textwidth / 2
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
            ],
            "font.size": font_size_main,
            "axes.labelsize": font_size_tex,  # Use the same font size as the main text.
            "axes.titlesize": font_size_tex,  # Not used - figure captions are handled by LaTeX \caption{}
            "axes.linewidth": line_width_axis,  # <= 0.8
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "xtick.labelsize": font_size_main,
            "xtick.major.size": 5 * 0.45 * line_width_axis,
            "xtick.major.width": 0.45 * line_width_axis,
            "xtick.major.pad": 0.15 * font_size_main,
            "ytick.labelsize": font_size_main,
            "ytick.major.size": 5 * 0.45 * line_width_axis,
            "ytick.major.width": 0.45 * line_width_axis,
            "ytick.major.pad": 0.15 * font_size_main,
            "grid.linestyle": "--",
            "grid.color": "#bdbdbd",
            "grid.linewidth": 0.35 * line_width_axis,
            "grid.alpha": 0.3,  # Transparency
            "lines.linewidth": 1.8 * line_width_axis,
            "lines.markersize": 3.3 * 1.8 * line_width_axis,  # linear
            "legend.fontsize": font_size_main,
            "legend.frameon": True,
            "legend.fancybox": True,
            "legend.framealpha": 0.6,
            "legend.edgecolor": "#bdbdbd",
            "patch.linewidth": line_width_axis,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0,
        }
    )
