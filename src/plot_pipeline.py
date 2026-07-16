"""Walk through the pipeline for one source at +40 degrees.

Three stages on one figure:
    1. the two microphone input signals near signal onset (mic 1 leads),
    2. the GCC-PHAT cross-correlation with its peak at the TDOA,
    3. the arcsin conversion from TDOA to angle, with numbers plugged in.

Output: plots/pipeline_plus40.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gcc_phat import SPEED_OF_SOUND, gcc_phat, tdoa_to_angle
from run_sim import (
    ARRAY_CENTER,
    FS,
    MIC_DISTANCE,
    PLOTS_DIR,
    SOURCE_RADIUS,
    simulate_capture,
)

TRUE_ANGLE = 40.0

COL_MIC0 = "#2a78d6"
COL_MIC1 = "#008300"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRIDLINE, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(BASELINE)


def main():
    rng = np.random.default_rng(42)
    signals = simulate_capture(TRUE_ANGLE, "anechoic", rng)

    # Run the actual pipeline
    max_tau = MIC_DISTANCE / SPEED_OF_SOUND
    tau, cc = gcc_phat(signals[0], signals[1], FS, max_tau=max_tau, interp=16)
    est_angle = tdoa_to_angle(tau, MIC_DISTANCE)

    # Locate signal onset empirically (the simulated RIR includes a
    # fractional-delay filter offset, so geometric arrival times are shifted).
    # Mic 1 is closer for a +40 deg source, so its onset comes first.
    onset1 = int(np.argmax(np.abs(signals[1]) > 0.15 * np.abs(signals[1]).max()))
    t_onset1 = onset1 / FS
    t_onset0 = t_onset1 + tau  # mic 0 lags by the measured TDOA

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7.6))
    fig.patch.set_facecolor(SURFACE)

    # --- Stage 1: input signals near onset -------------------------------
    start = onset1 - 8
    n_win = 56
    t_ms = (np.arange(start, start + n_win) / FS) * 1e3
    ax1.plot(t_ms, signals[1][start:start + n_win], color=COL_MIC1,
             linewidth=2, marker="o", markersize=3.5, label="mic 1 (closer)")
    ax1.plot(t_ms, signals[0][start:start + n_win], color=COL_MIC0,
             linewidth=2, marker="o", markersize=3.5, label="mic 0 (farther)")
    ax1.axvline(t_onset1 * 1e3, color=COL_MIC1, linewidth=1, linestyle="--", alpha=0.6)
    ax1.axvline(t_onset0 * 1e3, color=COL_MIC0, linewidth=1, linestyle="--", alpha=0.6)
    ymax = ax1.get_ylim()[1]
    ax1.annotate(
        "", xy=(t_onset0 * 1e3, ymax * 0.85), xytext=(t_onset1 * 1e3, ymax * 0.85),
        arrowprops=dict(arrowstyle="->", color=INK_SECONDARY, linewidth=1.2),
    )
    ax1.text((t_onset0 + t_onset1) / 2 * 1e3, ymax * 0.95,
             f"Δt = {tau * 1e6:.0f} µs", ha="center", fontsize=9,
             color=INK_SECONDARY)
    ax1.set_xlabel("time (ms)", color=INK_SECONDARY)
    ax1.set_ylabel("amplitude", color=INK_SECONDARY)
    ax1.set_title(
        f"1 — Input signals at the two mics (source at +{TRUE_ANGLE:.0f}°, "
        "onset zoom: same waveform, mic 0 delayed)",
        color=INK_PRIMARY, fontsize=10, loc="left",
    )
    leg = ax1.legend(loc="lower right", fontsize=9, framealpha=0.9)
    for text in leg.get_texts():
        text.set_color(INK_SECONDARY)
    style_axis(ax1)

    # --- Stage 2: GCC-PHAT cross-correlation -----------------------------
    max_shift = (len(cc) - 1) // 2
    lags_us = np.arange(-max_shift, max_shift + 1) / (16 * FS) * 1e6
    ax2.plot(lags_us, cc, color=COL_MIC0, linewidth=1.5)
    ax2.axvline(tau * 1e6, color=INK_SECONDARY, linewidth=1, linestyle="--")
    peak = cc[np.argmax(np.abs(cc))]
    ax2.plot([tau * 1e6], [peak], "o", color=COL_MIC1, markersize=8)
    ax2.annotate(
        f"peak → τ = {tau * 1e6:.0f} µs",
        xy=(tau * 1e6, peak), xytext=(tau * 1e6 - 320, peak * 0.9),
        fontsize=10, color=INK_PRIMARY,
        arrowprops=dict(arrowstyle="->", color=INK_SECONDARY, linewidth=1),
    )
    ax2.set_xlabel("lag between mic 0 and mic 1 (µs)", color=INK_SECONDARY)
    ax2.set_ylabel("GCC-PHAT correlation", color=INK_SECONDARY)
    ax2.set_title(
        "2 — GCC-PHAT cross-correlation over physically possible lags "
        f"(±d/c = ±{max_tau * 1e6:.0f} µs)",
        color=INK_PRIMARY, fontsize=10, loc="left",
    )
    style_axis(ax2)

    # --- Stage 3: TDOA -> angle, numbers plugged in ----------------------
    formula = (
        "3 — Angle:   θ = arcsin(c·τ / d) "
        f"= arcsin(343 m/s × {tau * 1e6:.0f} µs / {MIC_DISTANCE} m) "
        f"= arcsin({SPEED_OF_SOUND * tau / MIC_DISTANCE:.3f}) "
        f"=  +{est_angle:.1f}°     (true: +{TRUE_ANGLE:.0f}°)"
    )
    fig.text(0.5, 0.015, formula, ha="center", fontsize=10.5, color=INK_PRIMARY,
             bbox=dict(boxstyle="round,pad=0.55", facecolor=SURFACE,
                       edgecolor=BASELINE))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = os.path.join(PLOTS_DIR, "pipeline_plus40.png")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")
    expected_tau = MIC_DISTANCE * np.sin(np.radians(TRUE_ANGLE)) / SPEED_OF_SOUND
    print(f"tau = {tau * 1e6:.1f} us (geometric truth {expected_tau * 1e6:.1f} us), "
          f"estimated angle = {est_angle:.2f} deg")


if __name__ == "__main__":
    main()
