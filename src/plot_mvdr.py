"""Plot the MVDR spatial spectrum for a +40 degree source.

Same simulated capture as plot_beamformer.py. Shows MVDR in anechoic and
reverberant conditions, with the anechoic delay-and-sum response as a
reference to visualize MVDR's sharper main lobe. Power in dB.

Output: plots/mvdr_plus40.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from beamformer import estimate_angle_das
from mvdr import estimate_angle_mvdr
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


def to_db(spectrum):
    s = spectrum / spectrum.max()
    return 10 * np.log10(np.maximum(s, 1e-12))


def main():
    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    results = {}
    for scenario, color in (("anechoic", COL_ANECHOIC), ("reverberant", COL_REVERB)):
        signals = simulate_capture(TRUE_ANGLE, scenario, rng)
        est, angles, spectrum = estimate_angle_mvdr(signals, FS, MIC_DISTANCE)
        results[scenario] = est
        ax.plot(angles, to_db(spectrum), color=color, linewidth=2,
                label=f"MVDR, {scenario}")
        if scenario == "anechoic":
            das_est, das_angles, das_powers = estimate_angle_das(
                signals, FS, MIC_DISTANCE)
            ax.plot(das_angles, to_db(das_powers), color=INK_MUTED,
                    linewidth=1.5, linestyle="--",
                    label=f"delay-and-sum, anechoic (peak {das_est:+.1f}°)")
        peak_idx = int(np.argmax(spectrum))
        ax.plot([angles[peak_idx]], [0.0], "o", color=color, markersize=8)
        ax.annotate(
            f"peak {est:+.1f}°",
            xy=(angles[peak_idx], 0.0),
            xytext=(angles[peak_idx] - 48, 0.6 if scenario == "anechoic" else -2.6),
            fontsize=9.5, color=INK_SECONDARY,
            arrowprops=dict(arrowstyle="->", color=INK_MUTED, linewidth=1),
        )

    ax.axvline(TRUE_ANGLE, color=INK_SECONDARY, linewidth=1, linestyle="--")
    ax.text(TRUE_ANGLE + 1.5, ax.get_ylim()[0] * 0.95, f"true +{TRUE_ANGLE:.0f}°",
            fontsize=9.5, color=INK_SECONDARY, rotation=90, va="bottom")

    ax.set_xlim(-92, 92)
    ax.set_xticks(np.arange(-90, 91, 30))
    ax.grid(color=GRIDLINE, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(BASELINE)
    ax.set_xlabel("look direction (deg)", color=INK_SECONDARY)
    ax.set_ylabel("normalized power (dB)", color=INK_SECONDARY)
    ax.set_title(
        "MVDR (Capon) spatial spectrum, source at +40° "
        f"(2 mics, {MIC_DISTANCE} m spacing)",
        color=INK_PRIMARY, fontsize=11,
    )
    leg = ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    for text in leg.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "mvdr_plus40.png")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")
    for scenario, est in results.items():
        print(f"{scenario:12s} MVDR peak = {est:+.1f} deg (true +40.0)")


if __name__ == "__main__":
    main()
