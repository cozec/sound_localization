"""Plot the delay-and-sum beamformer response for a +40 degree source.

Steers the 2-mic array across -90..+90 degrees for the same simulated
capture used in plot_pipeline.py and plots normalized output power vs
steering angle, in anechoic and reverberant conditions.

Output: plots/beamformer_plus40.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from beamformer import estimate_angle_das
from run_sim import FS, MIC_DISTANCE, PLOTS_DIR, simulate_capture

TRUE_ANGLE = 40.0

COL_ANECHOIC = "#2a78d6"
COL_REVERB = "#008300"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"


def main():
    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    results = {}
    for scenario, color in (("anechoic", COL_ANECHOIC), ("reverberant", COL_REVERB)):
        signals = simulate_capture(TRUE_ANGLE, scenario, rng)
        est, angles, powers = estimate_angle_das(signals, FS, MIC_DISTANCE)
        norm = powers / powers.max()
        results[scenario] = est
        ax.plot(angles, norm, color=color, linewidth=2, label=scenario)
        peak_idx = int(np.argmax(norm))
        ax.plot([angles[peak_idx]], [1.0], "o", color=color, markersize=8)
        ax.annotate(
            f"peak {est:+.1f}°",
            xy=(angles[peak_idx], 1.0),
            xytext=(angles[peak_idx] - 52, 1.035 if scenario == "anechoic" else 0.90),
            fontsize=9.5, color=INK_SECONDARY,
            arrowprops=dict(arrowstyle="->", color=INK_MUTED, linewidth=1),
        )

    ax.axvline(TRUE_ANGLE, color=INK_SECONDARY, linewidth=1, linestyle="--")
    ax.text(TRUE_ANGLE + 1.5, 0.08, f"true +{TRUE_ANGLE:.0f}°",
            fontsize=9.5, color=INK_SECONDARY, rotation=90, va="bottom")

    ax.set_xlim(-92, 92)
    ax.set_ylim(0, 1.09)
    ax.set_xticks(np.arange(-90, 91, 30))
    ax.grid(color=GRIDLINE, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(BASELINE)
    ax.set_xlabel("steering angle (deg)", color=INK_SECONDARY)
    ax.set_ylabel("normalized beam output power", color=INK_SECONDARY)
    ax.set_title(
        "Delay-and-sum beamformer response, source at +40° "
        f"(2 mics, {MIC_DISTANCE} m spacing, white noise)",
        color=INK_PRIMARY, fontsize=11,
    )
    leg = ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    for text in leg.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "beamformer_plus40.png")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")
    for scenario, est in results.items():
        print(f"{scenario:12s} delay-and-sum peak = {est:+.1f} deg (true +40.0)")


if __name__ == "__main__":
    main()
