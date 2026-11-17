"""Plots of layouts and metrics in the paper's figure style. Requires the ``plot`` extra.

Style follows the paper figures: Linux Libertine (the ACM body font), fixed colors per KG
similarity measure, light grid, kNN as a dotted reference line, 200 dpi on white.
Korean class names fall back to a Korean font glyph by glyph.
"""

from __future__ import annotations

import numpy as np

# first font that has a glyph wins (matplotlib falls back per glyph)
FONTS = ["Linux Libertine O", "NanumGothic", "AppleGothic", "Noto Sans CJK KR", "DejaVu Sans"]
MEASURE_COLORS = {"jaccard": "#1f77b4", "overlap": "#2ca02c", "idf_cosine": "#d62728", "adamic_adar": "#9467bd"}
DPI = 200


def use_fonts() -> None:
    """Use the installed fonts of ``FONTS`` in that order (missing ones are skipped)."""
    import matplotlib as mpl
    from matplotlib import font_manager as fm

    installed = {f.name for f in fm.fontManager.ttflist}
    fonts = [f for f in FONTS if f in installed] or ["DejaVu Sans"]
    mpl.rcParams["font.family"] = fonts
    mpl.rcParams["mathtext.fontset"] = "custom"
    for k in ("mathtext.rm", "mathtext.it", "mathtext.bf"):
        mpl.rcParams[k] = fonts[0]
    mpl.rcParams["axes.unicode_minus"] = False


def save(fig, path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")


def _measures(metrics: dict) -> list[str]:
    return [m for m in metrics if m not in ("settings", "knn")]


def _top_legend(fig, axes, y: float = 1.02, ncol: int | None = None) -> None:
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=ncol or len(labels), fontsize=8.5, frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, y))


def scatter(
    Y: np.ndarray,
    labels: np.ndarray,
    *,
    ax=None,
    title: str | None = None,
    s: float = 2.5,
    class_names: list[str] | None = None,
):
    """Scatter ``Y`` colored by class (tab20 up to 20 classes). Returns the axes.

    With ``class_names``, a legend of the classes is drawn to the right of the axes.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if ax is None:
        _, ax = plt.subplots(figsize=(2.5, 2.5))
    n = int(np.max(labels)) + 1
    cmap = plt.get_cmap("tab20" if n <= 20 else "gist_ncar", n)
    ax.scatter(Y[:, 0], Y[:, 1], c=labels, cmap=cmap, vmin=-0.5, vmax=n - 0.5, s=s, alpha=0.6, linewidths=0)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
    if title:
        ax.set_title(title, fontsize=10)
    if class_names:
        handles = [Line2D([], [], marker="o", ls="", ms=4, color=cmap(i), label=class_names[i])
                   for i in range(min(n, len(class_names)))]
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6.5, frameon=False)
    return ax


def grid(panels: list[tuple[str, np.ndarray]], labels: np.ndarray, *, class_names: list[str] | None = None,
         suptitle: str | None = None):
    """Layouts side by side (paper Figure 3: kNN and three guidance strengths). Returns the figure."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(panels), figsize=(2.25 * len(panels), 2.5), squeeze=False)
    for i, (ax, (title, Y)) in enumerate(zip(axes[0], panels)):
        scatter(Y, labels, ax=ax, title=title, class_names=class_names if i == len(panels) - 1 else None)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, y=1.02)
    fig.tight_layout()
    return fig


def alpha_sweep(metrics: dict, *, dataset: str = "", encoder: str = ""):
    """NMI over LT_2D across guidance strength (one column of paper Figure 2): a line per measure,
    kNN (α=0) dotted. Returns the figure."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(2.9, 2.05 * 2), squeeze=False)
    axes = axes[:, 0]
    for r, (ax, key, label) in enumerate(zip(axes, ("nmi", "lt_2d"), ("NMI", r"$\mathrm{LT}_{2D}$"))):
        for m in _measures(metrics):
            alphas = sorted(metrics[m], key=float)
            ax.plot([float(a) for a in alphas], [metrics[m][a][key] for a in alphas], marker="o", ms=2.4,
                    lw=1.2, color=MEASURE_COLORS.get(m), label=m)
        ax.axhline(metrics["knn"][key], color="#222222", ls=":", lw=1.0, label="kNN (α=0)")
        ax.grid(alpha=0.3, lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7.5)
        ax.set_ylabel(label, fontsize=9)
        if r == len(axes) - 1:
            ax.set_xlabel("α", fontsize=9)
        if encoder and r == 0:  # once per column, as in the paper
            ax.text(0.03, 0.96, encoder, fontsize=8, style="italic", ha="left", va="top",
                    transform=ax.transAxes, color="#333333")
    if dataset:
        axes[0].set_title(dataset, fontsize=10, fontweight="bold", pad=6)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=min(len(labels), 3), fontsize=8, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), columnspacing=1.2, handlelength=2.0)
    return fig


def retention(metrics: dict, *, dataset: str = "", encoder: str = ""):
    """Fraction of the original kNN neighbors kept across guidance strength (paper Figure 4)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(2.9, 3.2))
    fig.subplots_adjust(left=0.2, right=0.96, bottom=0.14, top=0.68)  # header: title, then legend
    ax.axhline(1.0, color="gray", ls=":", lw=0.9)
    last = {}
    for m in _measures(metrics):
        alphas = sorted(metrics[m], key=float)
        ys = [metrics[m][a]["retention"] for a in alphas]
        ax.plot([float(a) for a in alphas], ys, marker="o", ms=2.8, lw=1.3, color=MEASURE_COLORS.get(m), label=m)
        last[m] = (float(alphas[-1]), ys[-1])
    if last:
        m, (x, y) = min(last.items(), key=lambda kv: kv[1][1])
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(-3, -13), fontsize=8.5, ha="right",
                    fontweight="bold", color=MEASURE_COLORS.get(m))
    ax.set_ylim(0.0, 1.08)
    ax.grid(alpha=0.3, lw=0.5)
    ax.tick_params(labelsize=7.5)
    ax.set_xlabel("α", fontsize=9)
    ax.set_ylabel("retention", fontsize=8.5)
    if dataset:
        ax.set_title(dataset, fontsize=10, fontweight="bold", pad=6)
    if encoder:
        ax.text(0.03, 0.06, encoder, fontsize=8, style="italic", transform=ax.transAxes, color="#333333")
    fig.suptitle("Fraction of original embedding-kNN neighbors retained\n(1.0 = no KG intervention)",
                 fontsize=9, y=0.98, va="top")
    _top_legend(fig, [ax], y=0.76, ncol=2)
    return fig
