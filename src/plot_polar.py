"""Polar beampatterns of the 2-mic array: delay-and-sum vs MVDR.

Left: closed-form delay-and-sum pattern |aᴴ(θ_look) a(θ)| / M steered to
+40°, at several frequencies, showing the beam narrowing with frequency,
grating lobes above the spatial-aliasing limit c / (2d), and the
front/back ambiguity of a linear array (pattern mirrored about the mic
axis at ±90°).

Right: MVDR pattern |wᴴ a(θ)| with w = Rₙ⁻¹a / (aᴴRₙ⁻¹a), where Rₙ is the
per-bin covariance of a simulated anechoic capture of an interferer at
−30°. Unit gain toward +40° is kept while a null is placed on the
interferer — the extra freedom the DOA estimate buys you. (Frequencies
where the +40° and −30° steering vectors coincide modulo 2π — e.g. 1 kHz
already has a delay-and-sum null at −30°, and at 2 kHz the two directions
are indistinguishable — are avoided so the comparison is meaningful.)

Output: plots/polar_patterns.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft

from gcc_phat import SPEED_OF_SOUND
from run_sim import FS, MIC_DISTANCE, PLOTS_DIR, simulate_capture

LOOK_ANGLE = 40.0
INTERFERER_ANGLE = -30.0
DAS_FREQS = [500, 1000, 2000, 4000]
MVDR_FREQS = [800, 1500]
FLOOR_DB = -30.0
CEIL_DB = 6.0   # MVDR gain off the look direction can exceed 0 dB

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"
PALETTE = ["#2a78d6", "#008300", "#d6612a", "#7a3fbf"]


def steering(theta_deg, freq):
    """Steering vectors (2, n_angles), mic 1 leading mic 0 for positive angles."""
    tau = MIC_DISTANCE * np.sin(np.radians(theta_deg)) / SPEED_OF_SOUND
    return np.vstack([np.ones_like(tau), np.exp(2j * np.pi * freq * tau)])


def to_db(pattern):
    return np.clip(20 * np.log10(np.maximum(pattern, 1e-12)), FLOOR_DB, CEIL_DB)


def das_pattern(theta_deg, freq):
    """Delay-and-sum magnitude response steered to LOOK_ANGLE."""
    w = steering(np.array([LOOK_ANGLE]), freq)[:, 0] / 2
    return np.abs(w.conj() @ steering(theta_deg, freq))


def mvdr_pattern(theta_deg, freq, X, freqs, loading=1e-2):
    """MVDR magnitude response using the covariance of the nearest STFT bin."""
    k = int(np.argmin(np.abs(freqs - freq)))
    Xk = X[:, k, :]
    R = (Xk @ Xk.conj().T) / Xk.shape[1]
    R = R + loading * (np.trace(R).real / 2) * np.eye(2)
    Rinv = np.linalg.inv(R)
    a = steering(np.array([LOOK_ANGLE]), freqs[k])[:, 0]
    w = Rinv @ a / (a.conj() @ Rinv @ a)
    return np.abs(w.conj() @ steering(theta_deg, freqs[k]))


def style_polar(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_theta_zero_location("N")     # broadside (+y) at the top
    ax.set_theta_direction(-1)          # positive angles toward +x mic (clockwise)
    ax.set_thetagrids(np.arange(0, 360, 30),
                      [f"{((a + 180) % 360) - 180:+d}°" for a in range(0, 360, 30)])
    ax.set_rlim(FLOOR_DB, CEIL_DB)
    ax.set_rticks([-20, -10, 0])
    ax.set_rlabel_position(200)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    ax.grid(color=GRIDLINE)
    ax.set_title(title, color=INK_PRIMARY, fontsize=11, pad=16)
    # Mic axis and look direction as reference lines.
    ax.plot([np.pi / 2, -np.pi / 2], [0, 0], color=GRIDLINE, linewidth=4, zorder=0)
    ax.plot([np.radians(LOOK_ANGLE)] * 2, [FLOOR_DB, 0], color=INK_SECONDARY,
            linewidth=1, linestyle=":", zorder=1)


def main():
    theta = np.linspace(-180, 180, 721)
    rad = np.radians(theta)

    fig, (ax_das, ax_mvdr) = plt.subplots(
        1, 2, figsize=(12, 6), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor(SURFACE)

    alias_hz = SPEED_OF_SOUND / (2 * MIC_DISTANCE)
    style_polar(ax_das, f"Delay-and-sum steered to {LOOK_ANGLE:+.0f}°\n"
                        f"d = {MIC_DISTANCE*100:.0f} cm, aliasing above "
                        f"{alias_hz:.0f} Hz")
    for f, color in zip(DAS_FREQS, PALETTE):
        ax_das.plot(rad, to_db(das_pattern(theta, f)), color=color,
                    linewidth=1.8, label=f"{f} Hz")
    ax_das.legend(loc="lower left", bbox_to_anchor=(-0.15, -0.12), fontsize=8,
                  frameon=False)

    rng = np.random.default_rng(42)
    noise = simulate_capture(INTERFERER_ANGLE, "anechoic", rng)
    freqs, _, X = stft(noise, fs=FS, nperseg=256)
    style_polar(ax_mvdr, f"MVDR steered to {LOOK_ANGLE:+.0f}°\n"
                         f"interferer at {INTERFERER_ANGLE:+.0f}° (null)")
    for f, color in zip(MVDR_FREQS, PALETTE[1:]):
        ax_mvdr.plot(rad, to_db(mvdr_pattern(theta, f, X, freqs)), color=color,
                     linewidth=1.8, label=f"MVDR {f} Hz")
        ax_mvdr.plot(rad, to_db(das_pattern(theta, f)), color=color,
                     linewidth=1, linestyle="--", alpha=0.6,
                     label=f"delay-and-sum {f} Hz")
    ax_mvdr.plot([np.radians(INTERFERER_ANGLE)] * 2, [FLOOR_DB, 0],
                 color="#b00020", linewidth=1, linestyle=":", zorder=1)
    ax_mvdr.legend(loc="lower left", bbox_to_anchor=(-0.15, -0.12), fontsize=8,
                   frameon=False)

    fig.text(0.5, 0.01,
             "Gain in dB, 0 dB = unit gain at the look direction. Grey bar = "
             "mic axis; linear arrays are mirror-symmetric about it "
             "(front/back ambiguity).",
             ha="center", color=INK_SECONDARY, fontsize=8)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, "polar_patterns.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
