"""US 2018/0176679 A1 end to end with the Fig. 6 array and real speech.

Array (patent Fig. 6): three microphones equidistant on a circle plus one at
the centre. Two talkers in a reverberant room (pyroomacoustics), rendered
separately so the target-only image at the centre mic is the clean reference:

    target speech     at TARGET_DEG   (0 dB)
    interferer speech at INTERF_DEG   (INTERF_DB re. target)
    + white sensor noise

Pipeline (patent Fig. 3 / Fig. 7):
    STFT (analysis filter bank) -> per-bin covariance R(f)
    -> DOA: MVDR scan over 0..360 deg averaged over the speech band
    -> steering direction delta = DOA estimate (+ a deliberate error)
    -> weights per bin:
         delay-and-sum   w = s(delta)/M
         MVDR            w = (R + δI)^-1 s / (s^H (R + δI)^-1 s),  δ = 1e-4 and 0.1 of mean power
         patent          W = R1^-1 A (A^H R1^-1 A)^-1 Delta,  A = [s(delta-w/2), s(delta), s(delta+w/2)]
                         R1 = R + eps I,  eps = c * (f/f_ref)^2 * E(f)
    -> ISTFT (synthesis filter bank)

Speech clips are synthesized with macOS `say` (two voices) into data/.
Steering vectors are free-field far-field for the circle (angle from +x,
counter-clockwise); the room adds reverberation the model does not know
about, which is the realistic mismatch case.

Output: plots/patent_circular_speech.png, results/patent_circular_*.wav
"""

import os
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
from scipy.signal import istft, stft

ROOT = os.path.join(os.path.dirname(__file__), "..")
PLOTS_DIR, DATA_DIR, RESULTS_DIR = (os.path.join(ROOT, d) for d in ("plots", "data", "results"))

FS = 16000
C = 343.0
RADIUS = 0.08                 # circle radius [m]; 3 mics on the ring + 1 centre
ROOM_DIMS = [6.0, 5.0]
CENTER = np.array([3.0, 2.0])
SOURCE_R = 1.5
TARGET_DEG, INTERF_DEG = 30.0, 150.0
INTERF_DB = -6.0              # interferer level re. target (so the DOA scan picks the target)
DOA_ERROR_DEG = 3.0           # deliberate steering error added to the DOA estimate
BEAM_WIDTH_DEG = 8.0          # pre-selected beam width (patent claim 5: < 10 deg)
NOISE_DB = -30.0              # sensor noise re. target level at mic 0
NPERSEG = 512
BAND = (300.0, 4000.0)
C_SLOPE, F_REF = 0.05, 2000.0

np.random.seed(0)


# --------------------------------------------------------------------------
# Speech
# --------------------------------------------------------------------------
def speech(name, voice, text):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        os.makedirs(DATA_DIR, exist_ok=True)
        subprocess.run(["say", "-v", voice, f"--data-format=LEI16@{FS}", "-o", path, text], check=True)
    fs, x = wavfile.read(path)
    assert fs == FS
    x = x.astype(float)
    return x / np.sqrt(np.mean(x**2))          # unit RMS


tgt = speech("speech_target.wav", "Samantha", "Turn on the living room lights and set the temperature to seventy degrees.")
itf = speech("speech_interferer.wav", "Daniel", "The quick brown fox jumps over the lazy dog while the rain keeps falling.")
n = min(len(tgt), len(itf)); tgt, itf = tgt[:n], itf[:n]

# --------------------------------------------------------------------------
# Array and room
# --------------------------------------------------------------------------
ring = np.radians([90, 210, 330])
MIC_XY = np.vstack([np.c_[RADIUS*np.cos(ring), RADIUS*np.sin(ring)], [[0.0, 0.0]]])   # (4, 2), last = centre
M = len(MIC_XY)


def render(angle_deg, sig):
    e_abs, max_order = pra.inverse_sabine(0.3, ROOM_DIMS)
    room = pra.ShoeBox(ROOM_DIMS, fs=FS, materials=pra.Material(e_abs), max_order=max_order)
    room.add_microphone_array(pra.MicrophoneArray((CENTER + MIC_XY).T, fs=FS))
    th = np.radians(angle_deg)
    room.add_source((CENTER + SOURCE_R*np.array([np.cos(th), np.sin(th)])).tolist(), signal=sig)
    room.simulate()
    return room.mic_array.signals[:, :n]


target_mics = render(TARGET_DEG, tgt)
interf_mics = 10**(INTERF_DB/20) * render(INTERF_DEG, itf)
lvl = np.sqrt(np.mean(target_mics[M - 1]**2))
noise = 10**(NOISE_DB/20) * lvl * np.random.randn(M, n)
X_time = target_mics + interf_mics + noise
REF_MIC = M - 1                                             # centre mic: steering vectors are relative to it
ref = target_mics[REF_MIC]                                  # clean reference: target image at the centre mic (with reverb)


def steer(theta_deg, f):
    """Free-field steering vector for the circle, relative to the centre mic."""
    th = np.radians(theta_deg)
    tau = -(MIC_XY[:, 0]*np.cos(th) + MIC_XY[:, 1]*np.sin(th)) / C        # arrival delay relative to centre
    return np.exp(-2j*np.pi*f*tau).reshape(-1, 1)


# --------------------------------------------------------------------------
# Analysis filter bank + per-bin covariance
# --------------------------------------------------------------------------
freqs, _, X = stft(X_time, fs=FS, nperseg=NPERSEG)         # (M, n_bins, n_frames)
bins = np.where((freqs >= BAND[0]) & (freqs <= BAND[1]))[0]
R = np.zeros((len(freqs), M, M), dtype=complex)
for k in range(len(freqs)):
    Xk = X[:, k, :]
    R[k] = (Xk @ Xk.conj().T) / Xk.shape[1]

# --------------------------------------------------------------------------
# Step 2: DOA — MVDR scan over the full circle, averaged over the band
# --------------------------------------------------------------------------
scan_deg = np.arange(0, 360, 1.0)
P = np.zeros(len(scan_deg))
for k in bins:
    Rinv = np.linalg.pinv(R[k] + 1e-3*np.trace(R[k]).real/M*np.eye(M))
    for i, th in enumerate(scan_deg):
        s = steer(th, freqs[k])
        P[i] += 1.0 / np.real(s.conj().T @ Rinv @ s).squeeze()
P /= len(bins)
P_db = 10*np.log10(P/P.max())
doa_est = scan_deg[np.argmax(P_db)]
delta = doa_est + DOA_ERROR_DEG
print(f"DOA scan peak: {doa_est:.0f}°  (true {TARGET_DEG:.0f}°); steering to {delta:.0f}° (+{DOA_ERROR_DEG:.0f}° deliberate error)")

# --------------------------------------------------------------------------
# Steps 3–5: weights per bin, apply, synthesis filter bank
# --------------------------------------------------------------------------
def weights(k, method):
    f = freqs[k]
    s = steer(delta, f)
    if method == "delay-and-sum":
        return s / M
    if method.startswith("MVDR"):
        dl = {"MVDR δ=1e-4": 1e-4, "MVDR δ=0.1": 0.1}[method]
        Rinv = np.linalg.pinv(R[k] + dl*np.trace(R[k]).real/M*np.eye(M))
        return (Rinv @ s) / (s.conj().T @ Rinv @ s)
    if method == "patent":
        A = np.hstack([steer(delta - BEAM_WIDTH_DEG/2, f), s, steer(delta + BEAM_WIDTH_DEG/2, f)])
        Delta = np.ones((3, 1))
        eps = C_SLOPE * (f/F_REF)**2 * np.trace(R[k]).real / M
        R1inv = np.linalg.pinv(R[k] + eps*np.eye(M))
        return R1inv @ A @ np.linalg.pinv(A.conj().T @ R1inv @ A) @ Delta
    raise ValueError(method)


METHODS = ["delay-and-sum", "MVDR δ=1e-4", "MVDR δ=0.1", "patent"]
outputs = {}
for m in METHODS:
    Y = np.zeros(X.shape[1:], dtype=complex)
    for k in range(len(freqs)):
        w = weights(k, m) if freqs[k] > 0 else steer(delta, 1.0) / M
        Y[k] = (w.conj().T @ X[:, k, :]).squeeze()
    _, y = istft(Y, fs=FS, nperseg=NPERSEG)
    outputs[m] = np.real(y[:n])


def snr_db(reference, est):
    # allow a scalar gain/phase fit so a pure level change is not counted as error
    g = np.dot(est, reference) / np.dot(est, est)
    e = reference - g*est
    return 10*np.log10(np.sum(reference**2) / np.sum(e**2))


results = {"centre mic (input)": snr_db(ref, X_time[REF_MIC])}
results.update({m: snr_db(ref, outputs[m]) for m in METHODS})
for k_, v in results.items():
    print(f"  {k_:16s} SNR {v:6.1f} dB")

os.makedirs(RESULTS_DIR, exist_ok=True)
for name, sig in [("input_centre_mic", X_time[REF_MIC]), ("target_clean", ref)] + [(m.replace("-", "_").replace(" ", "_").replace("δ=", "dl"), outputs[m]) for m in METHODS]:
    wavfile.write(os.path.join(RESULTS_DIR, f"patent_circular_{name}.wav"), FS, (0.5*sig/np.max(np.abs(sig))*32767).astype(np.int16))

# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def spec_db(x):
    _, _, S = stft(x, fs=FS, nperseg=NPERSEG)
    return 20*np.log10(np.abs(S) + 1e-6)


fig = plt.figure(figsize=(16, 11))
gs = fig.add_gridspec(3, 4, height_ratios=[1.1, 1, 1])

ax = fig.add_subplot(gs[0, 0])
ax.plot(MIC_XY[:3, 0]*100, MIC_XY[:3, 1]*100, "o", ms=10, label="ring mics")
ax.plot(0, 0, "s", ms=10, color="C3", label="centre mic")
circ = plt.Circle((0, 0), RADIUS*100, fill=False, ls="--", color="0.6"); ax.add_patch(circ)
for deg, c, lab in ((TARGET_DEG, "k", "target"), (INTERF_DEG, "r", "interferer")):
    ax.annotate("", xy=(14*np.cos(np.radians(deg)), 14*np.sin(np.radians(deg))), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=c, lw=1.5))
    ax.text(16*np.cos(np.radians(deg)), 16*np.sin(np.radians(deg)), f"{lab} {deg:.0f}°", color=c, ha="center", fontsize=9)
ax.set_aspect("equal"); ax.set_xlim(-20, 20); ax.set_ylim(-20, 20); ax.set_xlabel("cm"); ax.grid(alpha=0.3)
ax.set_title("Patent Fig. 6 array: 3 on a circle + centre", fontsize=11); ax.legend(fontsize=8, loc="lower right")

ax = fig.add_subplot(gs[0, 1], projection="polar")
ax.plot(np.radians(scan_deg), np.maximum(P_db, -20), lw=1.8)
ax.plot([np.radians(TARGET_DEG)]*2, [-20, 0], "k:", lw=1); ax.plot([np.radians(INTERF_DEG)]*2, [-20, 0], "r:", lw=1)
ax.set_theta_zero_location("E"); ax.set_theta_direction(1); ax.set_rlim(-20, 0); ax.set_rlabel_position(255)
ax.set_title(f"Step 2 — DOA scan (MVDR, {BAND[0]:.0f}–{BAND[1]:.0f} Hz avg): peak {doa_est:.0f}°", fontsize=11, pad=12)

ax = fig.add_subplot(gs[0, 2:])
names = list(results.keys()); vals = [results[k_] for k_ in names]
bars = ax.barh(names, vals, color=["0.6", "C0", "#c0392b", "#e67e22", "#1e8449"])
for b, v in zip(bars, vals):
    ax.text(v + 0.3, b.get_y() + b.get_height()/2, f"{v:.1f} dB", va="center", fontsize=10)
ax.set_xlabel("SNR vs clean target image at the centre mic [dB]"); ax.invert_yaxis(); ax.grid(alpha=0.3, axis="x")
ax.set_title(f"Steered to {delta:.0f}° = DOA estimate + {DOA_ERROR_DEG:.0f}° error, beam width {BEAM_WIDTH_DEG:.0f}°, RT60 0.3 s", fontsize=11)
ax.set_xlim(min(0, min(vals) - 2), max(vals) + 5)

t_axis = np.arange(n)/FS


def show_spec(ax, sig, title):
    S = spec_db(sig)
    ax.imshow(S, origin="lower", aspect="auto", extent=[0, n/FS, 0, FS/2], vmin=S.max()-70, vmax=S.max(), cmap="magma")
    ax.set_ylim(0, 5000); ax.set_title(title, fontsize=10); ax.set_xlabel("s"); ax.set_ylabel("Hz")


show_spec(fig.add_subplot(gs[1, 0]), ref, "target (clean image at centre mic)")
show_spec(fig.add_subplot(gs[1, 1]), X_time[REF_MIC], "centre mic input: target + interferer + noise")
show_spec(fig.add_subplot(gs[1, 2]), outputs["delay-and-sum"], "delay-and-sum output")
show_spec(fig.add_subplot(gs[1, 3]), outputs["MVDR δ=1e-4"], "MVDR (δ=1e-4) output")
show_spec(fig.add_subplot(gs[2, 0]), outputs["patent"], "patent output")

ax = fig.add_subplot(gs[2, 1:])
seg = slice(int(1.2*FS), int(1.5*FS))
ax.plot(t_axis[seg], ref[seg], "k", lw=1.2, label="clean target")
for m, c in (("MVDR δ=1e-4", "#c0392b"), ("patent", "#1e8449")):
    y = outputs[m]; g = np.dot(y, ref) / np.dot(y, y)          # gain-match for display
    ax.plot(t_axis[seg], g*y[seg], color=c, lw=0.9, alpha=0.9, label=f"{m} output")
ax.set_xlabel("s"); ax.set_title("Waveform detail, 1.2–1.5 s (outputs gain-matched to the reference)", fontsize=10)
ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="upper right")

fig.suptitle("US 2018/0176679 A1 with the Fig. 6 circular array and real speech — DOA scan, then beamform per STFT bin", fontsize=13, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.97))
os.makedirs(PLOTS_DIR, exist_ok=True)
out = os.path.join(PLOTS_DIR, "patent_circular_speech.png")
fig.savefig(out, dpi=120)
print(f"wrote {out}")
