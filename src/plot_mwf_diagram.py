"""Block diagram of the multichannel Wiener filter (as implemented).

Signal path: 2 mics -> STFT -> per-bin filtering Y = W^H X -> ISTFT.
Statistics path: X -> Rn (noise-only frames) and Rx (all frames) -> W.

Output: plots/mwf_diagram.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from run_sim import PLOTS_DIR

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
COL_SIGNAL = "#2a78d6"   # signal path
COL_STATS = "#008300"    # statistics path


def box(ax, cx, cy, w, h, text, edge=BASELINE, fontsize=9.5):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.08", facecolor=SURFACE,
        edgecolor=edge, linewidth=1.4))
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, color=INK_PRIMARY)


def arrow(ax, start, end, color, style="-|>", lw=1.6, connection="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=14,
        color=color, linewidth=lw, connectionstyle=connection,
        shrinkA=2, shrinkB=2))


def main():
    fig, ax = plt.subplots(figsize=(12, 6.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    Y_MAIN = 4.7   # signal path row
    Y_COV = 2.7    # covariance row
    Y_W = 1.1      # weight row

    # ---- signal path ------------------------------------------------------
    ax.text(0.55, Y_MAIN + 0.35, "x₀(t)", fontsize=11, color=INK_PRIMARY,
            ha="center")
    ax.text(0.55, Y_MAIN - 0.35, "x₁(t)", fontsize=11, color=INK_PRIMARY,
            ha="center")
    ax.text(0.55, Y_MAIN + 0.72, "2 mics", fontsize=8.5, color=INK_MUTED,
            ha="center")

    box(ax, 2.0, Y_MAIN, 1.3, 1.3, "STFT\n(256-sample\nframes)")
    arrow(ax, (0.95, Y_MAIN + 0.35), (1.32, Y_MAIN + 0.30), COL_SIGNAL)
    arrow(ax, (0.95, Y_MAIN - 0.35), (1.32, Y_MAIN - 0.30), COL_SIGNAL)

    # X(f,t) node
    ax.text(3.7, Y_MAIN + 0.30, "X(f, t)  ∈ ℂ²", fontsize=10.5,
            color=INK_PRIMARY, ha="center")
    arrow(ax, (2.70, Y_MAIN), (7.05, Y_MAIN), COL_SIGNAL)

    box(ax, 7.9, Y_MAIN, 1.7, 1.0, "per-bin filter\nY(f,t) = W(f)ᴴ X(f,t)",
        edge=COL_SIGNAL)
    box(ax, 10.15, Y_MAIN, 1.2, 0.9, "ISTFT")
    arrow(ax, (8.80, Y_MAIN), (9.50, Y_MAIN), COL_SIGNAL)
    arrow(ax, (10.80, Y_MAIN), (11.45, Y_MAIN), COL_SIGNAL)
    ax.text(11.62, Y_MAIN, "ŝ₀(t)", fontsize=11, color=INK_PRIMARY,
            ha="left", va="center")
    ax.text(11.35, Y_MAIN - 0.55, "target estimate\nat reference mic 0",
            fontsize=8.5, color=INK_MUTED, ha="center")

    # ---- statistics path --------------------------------------------------
    tap_x = 4.6
    arrow(ax, (tap_x, Y_MAIN - 0.06), (3.35, Y_COV + 0.55), COL_STATS,
          connection="arc3,rad=-0.25")
    arrow(ax, (tap_x, Y_MAIN - 0.06), (6.05, Y_COV + 0.55), COL_STATS,
          connection="arc3,rad=0.25")
    ax.plot([tap_x], [Y_MAIN], "o", color=COL_STATS, markersize=5)

    box(ax, 3.1, Y_COV, 2.6, 1.0,
        "noise covariance  Rₙ(f)\nframes with target silent\n(lead-in, t < 0.8 s)",
        edge=COL_STATS, fontsize=9)
    box(ax, 6.4, Y_COV, 2.2, 1.0,
        "mixture covariance  Rₓ(f)\nall frames",
        edge=COL_STATS, fontsize=9)

    box(ax, 4.75, Y_W, 3.9, 0.95,
        "W(f) = (Rₓ + δI)⁻¹ (Rₓ − Rₙ) e_ref",
        edge=COL_STATS, fontsize=10.5)
    arrow(ax, (3.1, Y_COV - 0.55), (3.7, Y_W + 0.52), COL_STATS)
    arrow(ax, (6.4, Y_COV - 0.55), (5.8, Y_W + 0.52), COL_STATS)

    # weights feed the filter (routed to the right of the Rx box)
    arrow(ax, (6.78, Y_W), (8.35, Y_MAIN - 0.58), COL_STATS,
          connection="arc3,rad=-0.42")
    ax.text(9.55, 2.05, "weights\n(one 2-vector per bin)", fontsize=8.5,
            color=INK_MUTED, ha="center")

    # ---- annotations ------------------------------------------------------
    ax.text(0.4, 0.35,
            "e_ref = [1, 0]ᵀ selects mic 0 as reference · δ = 1% diagonal loading · "
            "Rₓ − Rₙ estimates the target covariance Rₛ",
            fontsize=9, color=INK_SECONDARY, ha="left")
    ax.text(0.4, 0.0,
            "equivalent factorization: MWF = MVDR beamformer (spatial) × "
            "single-channel Wiener postfilter (spectral)",
            fontsize=9, color=INK_SECONDARY, ha="left")

    ax.text(1.1, 5.85, "signal path", fontsize=9, color=COL_SIGNAL)
    ax.plot([0.45, 1.0], [5.88, 5.88], color=COL_SIGNAL, linewidth=2)
    ax.text(3.0, 5.85, "statistics path", fontsize=9, color=COL_STATS)
    ax.plot([2.35, 2.9], [5.88, 5.88], color=COL_STATS, linewidth=2)

    ax.set_title("Multichannel Wiener filter — signal flow (src/wiener.py)",
                 color=INK_PRIMARY, fontsize=12, pad=12)

    out = os.path.join(PLOTS_DIR, "mwf_diagram.png")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
