"""Multichannel Wiener filter experiment: target at +40, interferer at -30.

Reverberant room (RT60 = 0.3 s). The target emits noise bursts after a
0.8 s silent lead-in; a continuous white-noise interferer plays throughout,
scaled to 0 dB SNR at the reference mic, plus weak sensor noise. The MWF
learns the noise covariance from the lead-in and is evaluated by filtering
the target and noise components separately with the same weights.

Outputs:
    plots/mwf_demo.png
    printed input/output SNR
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import stft

from run_sim import (
    ARRAY_CENTER,
    FS,
    MIC_DISTANCE,
    PLOTS_DIR,
    SOURCE_RADIUS,
    build_room,
)
from wiener import apply_weights, enhance

TARGET_ANGLE = 40.0
NOISE_ANGLE = -30.0
NOISE_LEAD_SECONDS = 0.8   # target-silent lead-in for noise cov estimation
DURATION = 3.0
NPERSEG = 256

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
COL_ENHANCED = "#2a78d6"
COL_CLEAN = "#008300"

CMAP_BLUES = LinearSegmentedColormap.from_list(
    "palette_blues",
    ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)


def make_target_signal(rng):
    """Noise bursts with a Hann envelope after a silent lead-in."""
    sig = np.zeros(int(DURATION * FS))
    burst_len = int(0.4 * FS)
    gap_len = int(0.3 * FS)
    pos = int(NOISE_LEAD_SECONDS * FS)
    while pos + burst_len <= len(sig):
        sig[pos:pos + burst_len] = (rng.standard_normal(burst_len)
                                    * np.hanning(burst_len))
        pos += burst_len + gap_len
    return sig


def render_source(angle_deg, signal, rng):
    """Simulate one source through the reverberant room; return mic signals."""
    room = build_room("reverberant", rng)
    mic_positions = np.array(
        [
            [ARRAY_CENTER[0] - MIC_DISTANCE / 2, ARRAY_CENTER[0] + MIC_DISTANCE / 2],
            [ARRAY_CENTER[1], ARRAY_CENTER[1]],
        ]
    )
    room.add_microphone_array(pra.MicrophoneArray(mic_positions, fs=FS))
    theta = np.radians(angle_deg)
    pos = ARRAY_CENTER + SOURCE_RADIUS * np.array([np.sin(theta), np.cos(theta)])
    room.add_source(pos.tolist(), signal=signal)
    room.simulate()
    return room.mic_array.signals


def spectrogram_db(x):
    freqs, times, S = stft(x, fs=FS, nperseg=NPERSEG)
    return freqs, times, 20 * np.log10(np.abs(S) + 1e-9)


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_MUTED, labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_color(BASELINE)


def main():
    rng = np.random.default_rng(42)

    target_mics = render_source(TARGET_ANGLE, make_target_signal(rng), rng)
    noise_mics = render_source(
        NOISE_ANGLE, rng.standard_normal(int(DURATION * FS)), rng)

    n = min(target_mics.shape[1], noise_mics.shape[1])
    target_mics, noise_mics = target_mics[:, :n], noise_mics[:, :n]
    noise_mics = noise_mics + 0.05 * np.std(noise_mics) * rng.standard_normal(
        noise_mics.shape)

    # Scale the noise for 0 dB SNR at the reference mic over the active period
    active = np.arange(n) / FS >= NOISE_LEAD_SECONDS + 0.1
    p_target = np.mean(target_mics[0, active] ** 2)
    noise_mics *= np.sqrt(p_target / np.mean(noise_mics[0, active] ** 2))
    mixed = target_mics + noise_mics

    enhanced, W = enhance(mixed, FS, NOISE_LEAD_SECONDS, nperseg=NPERSEG)

    # Component-wise evaluation with the same weights
    from scipy.signal import istft
    _, _, St = stft(target_mics, fs=FS, nperseg=NPERSEG)
    _, _, Sn = stft(noise_mics, fs=FS, nperseg=NPERSEG)
    _, yt = istft(apply_weights(W, St), fs=FS, nperseg=NPERSEG)
    _, yn = istft(apply_weights(W, Sn), fs=FS, nperseg=NPERSEG)
    yt, yn = yt[:n], yn[:n]

    snr_in = 10 * np.log10(p_target / np.mean(noise_mics[0, active] ** 2))
    snr_out = 10 * np.log10(np.mean(yt[active] ** 2)
                            / np.mean(yn[active] ** 2))
    print(f"input  SNR at mic 0 = {snr_in:5.1f} dB")
    print(f"output SNR (MWF)    = {snr_out:5.1f} dB   "
          f"(improvement {snr_out - snr_in:+.1f} dB)")

    # ---- Figure: input/output spectrograms + waveforms -------------------
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
        5, 1, figsize=(10, 12),
        gridspec_kw={"height_ratios": [1.6, 1.6, 1, 1, 1]})
    fig.patch.set_facecolor(SURFACE)

    f1, t1, in_db = spectrogram_db(mixed[0])
    vmax = in_db.max()
    im1 = ax1.pcolormesh(t1, f1 / 1e3, in_db, cmap=CMAP_BLUES,
                         vmin=vmax - 50, vmax=vmax, shading="auto")
    ax1.set_title("1 — Noisy mixture at mic 0 (target bursts at +40°, "
                  "interferer at −30°, 0 dB SNR)",
                  color=INK_PRIMARY, fontsize=10, loc="left")
    f2, t2, out_db = spectrogram_db(enhanced)
    im2 = ax2.pcolormesh(t2, f2 / 1e3, out_db, cmap=CMAP_BLUES,
                         vmin=vmax - 50, vmax=vmax, shading="auto")
    ax2.set_title(f"2 — MWF output (noise covariance from first "
                  f"{NOISE_LEAD_SECONDS:.1f} s, target silent)",
                  color=INK_PRIMARY, fontsize=10, loc="left")
    for ax, im in ((ax1, im1), (ax2, im2)):
        ax.set_ylabel("frequency (kHz)", color=INK_SECONDARY)
        ax.axvline(NOISE_LEAD_SECONDS, color=SURFACE, linewidth=1,
                   linestyle="--", alpha=0.9)
        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.set_label("dB", color=INK_SECONDARY, fontsize=8.5)
        cb.ax.tick_params(colors=INK_MUTED, labelsize=8)
        style_axis(ax)

    t = np.arange(n) / FS
    ylim = 1.1 * max(np.abs(mixed[0]).max(), np.abs(enhanced).max(),
                     np.abs(target_mics[0]).max())
    wave_panels = (
        (ax3, mixed[0], BASELINE,
         f"3 — Noisy mixture at mic 0 (SNR {snr_in:.1f} dB)"),
        (ax4, enhanced, COL_ENHANCED,
         f"4 — MWF output (SNR {snr_out:.1f} dB, {snr_out - snr_in:+.1f} dB)"),
        (ax5, target_mics[0], COL_CLEAN,
         "5 — Clean target at mic 0 (reference)"),
    )
    for ax, sig, color, title in wave_panels:
        ax.plot(t, sig, color=color, linewidth=0.6)
        ax.axvline(NOISE_LEAD_SECONDS, color=INK_MUTED, linewidth=1,
                   linestyle="--")
        ax.set_ylim(-ylim, ylim)
        ax.set_ylabel("amplitude", color=INK_SECONDARY)
        ax.set_title(title, color=INK_PRIMARY, fontsize=10, loc="left")
        ax.grid(color=GRIDLINE, linewidth=0.7)
        ax.set_axisbelow(True)
        style_axis(ax)
    ax3.text(NOISE_LEAD_SECONDS / 2, ylim * 0.65, "noise only\n(learn Rn)",
             ha="center", fontsize=8.5, color=INK_SECONDARY)
    ax5.set_xlabel("time (s)", color=INK_SECONDARY)

    fig.suptitle("Multichannel Wiener filter, 2-mic array (reverberant room)",
                 color=INK_PRIMARY, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = os.path.join(PLOTS_DIR, "mwf_demo.png")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
