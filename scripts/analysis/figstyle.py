"""One plotting style for every paper figure (single-column ICML width, colour-blind-safe Okabe-Ito palette)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PAL = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000")
W1, W2 = 3.25, 6.75  # inches: one column, two columns

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "savefig.bbox": "tight"})


def placeholder(path, text):
    fig, ax = plt.subplots(figsize=(W1, 1.6))
    ax.axis("off")
    ax.text(0.5, 0.5, "PLACEHOLDER\n" + text, ha="center", va="center", color="#D55E00")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
# Print style of paper/reviews/DIAGRAM_STRATEGY.md §1 (from the TDR scripts/style.py). Figures that call apply() are
# drawn at their printed size, saved with save() (no tight crop, so LaTeX includes them 1:1) and font-checked.
# The legacy rcParams above stay for figures that have not moved over yet.
MIN_PT = 6.5      # hard floor for any text at print size (asserted in save)
LABEL_PT = 7.0
TITLE_PT = 8.0

BLUE = "#2a78d6"       # ours; certify / commit; probe glyph
TEAL = "#0b6670"       # physics-likelihood inference without our gate
VIOLET = "#7b4fc4"     # capxfer (prior certificate)
INK = "#0b0b0b"        # oracle, identified-set bound, theory
INK2 = "#52514e"       # secondary ink, callouts
GREY = "#8a8985"       # baselines, refuse / abstain, context, foreign merge modules
TINT = "#F1F0ED"
GRID = "#e4e3df"
ORANGE = "#eb6834"     # ONLY unsafe commit, faulted wheel, T1->T2 flip, region above U*
ORANGE_TX = "#b8461a"  # text in orange contexts
SURFACE = "#ffffff"

LW_DATA, LW_OURS, LW_THEORY, LW_AXES, LW_GRID, LW_BOX = 1.0, 1.4, 0.9, 0.6, 0.4, 0.6
MS, CI_ALPHA = 3.5, 0.18
BOX_ROUND = 2 / 72     # 2 pt corner radius, inches
ARROW_SHRINK = 1.8     # pt: arrows stop short of their targets


def apply():
    matplotlib.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "STIX Two Text", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": LABEL_PT, "axes.titlesize": TITLE_PT, "axes.labelsize": LABEL_PT,
        "xtick.labelsize": LABEL_PT, "ytick.labelsize": LABEL_PT, "legend.fontsize": LABEL_PT,
        "axes.edgecolor": GREY, "axes.linewidth": LW_AXES, "axes.labelcolor": INK2, "xtick.color": INK2,
        "ytick.color": INK2, "xtick.major.width": LW_AXES, "ytick.major.width": LW_AXES,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "grid.color": GRID,
        "grid.linewidth": LW_GRID, "axes.axisbelow": True, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "hatch.linewidth": 0.5, "lines.linewidth": LW_DATA, "lines.markersize": MS, "pdf.fonttype": 42,
        "savefig.bbox": None, "savefig.pad_inches": 0})


def panel(ax, text, x=None, pad=3):
    """Panel title: bold 8 pt '(a) Sentence-case claim', left aligned, ink."""
    kw = dict(loc="left", fontsize=TITLE_PT, fontweight="bold", color=INK, pad=pad)
    if x is not None:
        kw["x"] = x
    return ax.set_title(text, **kw)


def check_fonts(fig, min_pt=MIN_PT):
    """Every visible text prints at >= min_pt (the figure is included 1:1)."""
    bad = [(t.get_text()[:30], round(t.get_fontsize(), 2)) for t in fig.findobj(matplotlib.text.Text)
           if t.get_visible() and t.get_text().strip() and t.get_fontsize() < min_pt - 1e-6]
    assert not bad, f"text under {min_pt} pt at print: {bad}"


def save(fig, path, png_dpi=None):
    """Save at the figure's exact size (no tight crop) after the font-floor check; optional PNG preview."""
    check_fonts(fig)
    path.parent.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0}):
        fig.savefig(path)
        if png_dpi:
            fig.savefig(path.with_suffix(".png"), dpi=png_dpi)
    plt.close(fig)
