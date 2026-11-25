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
