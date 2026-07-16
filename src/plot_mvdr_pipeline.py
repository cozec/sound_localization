"""Walk through the MVDR algorithm stage by stage for a +40 degree source.

Four panels:
    1. STFT magnitude of mic 0 — the time-frequency representation.
    2. Phase of the cross-covariance R01(f) vs frequency — the inter-mic
       delay lives here as a linear (wrapped) phase slope -2*pi*f*tau.
    3. Narrowband Capon pseudo-spectrum per frequency bin (angle x freq
       heatmap) — every bin votes for the source direction.
    4. Band-averaged MVDR spectrum -> peak -> angle estimate.

Output: plots/mvdr_pipeline_plus40.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import stft

from gcc_phat import SPEED_OF_SOUND
from mvdr import estimate_angle_mvdr
from run_sim import FS, MIC_DISTANCE, PLOTS_DIR, simulate_capture

TRUE_ANGLE = 40.0
NPERSEG = 256
BAND = (300.0, 7000.0)
LOADING = 1e-2

COL_MEASURED = "#2a78d6"
COL_THEORY = "#008300"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

# Sequential blue ramp from the palette (light -> dark) for magnitude maps.
CMAP_BLUES = LinearSegmentedColormap.from_list(
    "palette_blues",
    ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_MUTED, labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_color(BASELINE)


def main():
    rng = np.random.default_rng(42)
    signals = simulate_capture(TRUE_ANGLE, "anechoic", rng)

    freqs, times, X = stft(np.asarray(signals, dtype=float), fs=FS, nperseg=NPERSEG)
    bins = np.where((freqs >= BAND[0]) & (freqs <= BAND[1]))[0]
    expected_tau = MIC_DISTANCE * np.sin(np.radians(TRUE_ANGLE)) / SPEED_OF_SOUND

    # Per-bin covariance, cross-term phase, and narrowband Capon spectra
    scan = np.arange(-90.0, 90.5, 0.5)
    taus = MIC_DISTANCE * np.sin(np.radians(scan)) / SPEED_OF_SOUND
    eye = np.eye(2)
    cross_phase = np.empty(len(bins))
    narrowband = np.empty((len(bins), len(scan)))
    for row, k in enumerate(bins):
        Xk = X[:, k, :]
        R = (Xk @ Xk.conj().T) / Xk.shape[1]
        cross_phase[row] = np.angle(R[0, 1])
        R = R + LOADING * (np.trace(R).real / 2) * eye
        Rinv = np.linalg.inv(R)
        A = np.vstack([np.ones_like(taus), np.exp(2j * np.pi * freqs[k] * taus)])
        P = 1.0 / np.maximum(np.einsum("ia,ij,ja->a", A.conj(), Rinv, A).real, 1e-12)
        narrowband[row] = P / P.max()

    est, angles, spectrum = estimate_angle_mvdr(signals, FS, MIC_DISTANCE)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6))
    fig.patch.set_facecolor(SURFACE)
    ax1, ax2, ax3, ax4 = axes.flat

    # --- 1: STFT magnitude of mic 0 --------------------------------------
    mag_db = 20 * np.log10(np.abs(X[0]) + 1e-9)
    im1 = ax1.pcolormesh(times * 1e3, freqs / 1e3, mag_db, cmap=CMAP_BLUES,
                         vmin=mag_db.max() - 50, vmax=mag_db.max(), shading="auto")
    ax1.set_xlabel("time (ms)", color=INK_SECONDARY)
    ax1.set_ylabel("frequency (kHz)", color=INK_SECONDARY)
    ax1.set_title("1 — STFT of mic 0 (256-sample frames)",
                  color=INK_PRIMARY, fontsize=10, loc="left")
    cb1 = fig.colorbar(im1, ax=ax1, pad=0.02)
    cb1.set_label("magnitude (dB)", color=INK_SECONDARY, fontsize=8.5)
    cb1.ax.tick_params(colors=INK_MUTED, labelsize=8)
    style_axis(ax1)

    # --- 2: cross-covariance phase vs frequency --------------------------
    f_band = freqs[bins]
    predicted = np.angle(np.exp(-2j * np.pi * f_band * expected_tau))
    ax2.plot(f_band / 1e3, predicted, color=COL_THEORY, linewidth=2,
             label=f"theory: −2πf·τ(+40°), τ = {expected_tau * 1e6:.0f} µs")
    ax2.plot(f_band / 1e3, cross_phase, "o", color=COL_MEASURED, markersize=3,
             label="measured  arg R₀₁(f)")
    ax2.set_xlabel("frequency (kHz)", color=INK_SECONDARY)
    ax2.set_ylabel("phase (rad)", color=INK_SECONDARY)
    ax2.set_yticks([-np.pi, 0, np.pi], ["−π", "0", "π"])
    ax2.grid(color=GRIDLINE, linewidth=0.7)
    ax2.set_axisbelow(True)
    ax2.set_title("2 — Covariance cross-term phase arg R₀₁(f): "
                  "the TDOA as a phase slope",
                  color=INK_PRIMARY, fontsize=10, loc="left")
    leg = ax2.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
    for text in leg.get_texts():
        text.set_color(INK_SECONDARY)
    style_axis(ax2)

    # --- 3: narrowband Capon spectra (angle x frequency) ------------------
    im3 = ax3.pcolormesh(scan, f_band / 1e3, narrowband, cmap=CMAP_BLUES,
                         vmin=0, vmax=1, shading="auto")
    ax3.axvline(TRUE_ANGLE, color=SURFACE, linewidth=1, linestyle="--", alpha=0.9)
    ax3.set_xlabel("look direction (deg)", color=INK_SECONDARY)
    ax3.set_ylabel("frequency (kHz)", color=INK_SECONDARY)
    ax3.set_title("3 — Narrowband Capon spectrum 1/(aᴴR⁻¹a) per bin "
                  "(each row normalized)",
                  color=INK_PRIMARY, fontsize=10, loc="left")
    cb3 = fig.colorbar(im3, ax=ax3, pad=0.02)
    cb3.set_label("normalized power", color=INK_SECONDARY, fontsize=8.5)
    cb3.ax.tick_params(colors=INK_MUTED, labelsize=8)
    style_axis(ax3)

    # --- 4: band average -> peak -> angle ---------------------------------
    s_db = 10 * np.log10(spectrum / spectrum.max())
    ax4.plot(angles, s_db, color=COL_MEASURED, linewidth=2)
    peak_idx = int(np.argmax(spectrum))
    ax4.plot([angles[peak_idx]], [0.0], "o", color=COL_THEORY, markersize=8)
    ax4.axvline(TRUE_ANGLE, color=INK_SECONDARY, linewidth=1, linestyle="--")
    ax4.annotate(
        f"peak → θ̂ = {est:+.1f}°   (true +{TRUE_ANGLE:.0f}°)",
        xy=(angles[peak_idx], 0.0), xytext=(-70, -2.5),
        fontsize=10, color=INK_PRIMARY,
        arrowprops=dict(arrowstyle="->", color=INK_SECONDARY, linewidth=1),
    )
    ax4.set_xlabel("look direction (deg)", color=INK_SECONDARY)
    ax4.set_ylabel("normalized power (dB)", color=INK_SECONDARY)
    ax4.set_xticks(np.arange(-90, 91, 30))
    ax4.grid(color=GRIDLINE, linewidth=0.7)
    ax4.set_axisbelow(True)
    ax4.set_title("4 — Average bins 0.3–7 kHz → broadband MVDR spectrum → argmax",
                  color=INK_PRIMARY, fontsize=10, loc="left")
    style_axis(ax4)

    fig.suptitle("MVDR pipeline, source at +40°: STFT → per-bin covariance → "
                 "Capon spectra → band average → angle",
                 color=INK_PRIMARY, fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(PLOTS_DIR, "mvdr_pipeline_plus40.png")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")
    print(f"estimated angle = {est:+.2f} deg (true +{TRUE_ANGLE:.0f})")


if __name__ == "__main__":
    main()
