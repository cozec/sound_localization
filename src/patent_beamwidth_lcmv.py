"""Demo of US 2018/0176679 A1 (Verizon): beamforming with a pre-selected
beam width and transition slope, standardized across frequency.

The patent's beamformer is LCMV with a steering *matrix*
    A(f, δ) = [ s(f, δ − w/2),  s(f, δ),  s(f, δ + w/2) ]      (boundary vectors)
constrained to A^H W = Δ, plus a loading term ε(f, n) = c(slope) · b(f) · E(n)
that scales with frequency and input power:
    J = W^H R W + ε W^H W − 2 λ^H (A^H W − Δ)
    W = R1^-1 A (A^H R1^-1 A)^-1 Δ,   R1 = R + ε I
The boundary vectors keep unit gain across the whole beam width, so a
target a couple of degrees off the estimated DOA is not self-nulled; b(f)
equalizes the beam width over frequency.

Physical mic array here: 16 mics, 4 cm spacing (aliasing above 4.3 kHz),
c = 343 m/s, narrowband snapshots per frequency. Target at 20°, DOA estimate
22° (2° error), interferer at −40° at +10 dB. Compared against MVDR with
no loading and MVDR with constant diagonal loading.

Note on the patent's 5 deg example: forcing a -3 dB edge 2.5 deg from the
look direction is far narrower than this array can form (natural width
10-70 deg over 0.5-4 kHz) and drives the weights superdirective (+20 dB
sidelobes). The flat-top form Delta = [1, 1, 1] is what is demonstrated:
unit gain held across the width, natural roll-off outside, at the cost of a
few dB of sidelobe bulge next to the beam.

Output: plots/patent_beamwidth_lcmv.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")

np.random.seed(0)
C_SOUND = 343.0
M = 16
D_M = 0.04                 # mic spacing [m]
N_SNAP = 4000
TARGET_DEG = 20.0
DOA_EST_DEG = 22.0         # estimated DOA, 2 deg off
INTERF_DEG = -40.0
INTERF_DB = 10.0
NOISE = 0.3
BEAM_WIDTH_DEG = 5.0       # patent example: +-2.5 deg boundary vectors
FREQS = np.array([500, 1000, 1500, 2000, 2500, 3000, 3500, 4000])
F_SHOW = 2000


def steer(theta_deg, f):
    tau = D_M * np.arange(M) * np.sin(np.radians(theta_deg)) / C_SOUND
    return np.exp(2j * np.pi * f * tau).reshape(-1, 1)


def simulate(f, target_amp=1.0):
    """Narrowband snapshots at frequency f: X = s_t a_t + s_i a_i + noise."""
    a_t = target_amp * np.exp(1j * 2*np.pi*np.random.rand(N_SNAP))            # unit-modulus random phase
    a_i = 10**(INTERF_DB/20) * np.exp(1j * 2*np.pi*np.random.rand(N_SNAP))
    X = steer(TARGET_DEG, f) @ a_t.reshape(1, -1) + steer(INTERF_DEG, f) @ a_i.reshape(1, -1)
    X += NOISE * (np.random.randn(M, N_SNAP) + 1j*np.random.randn(M, N_SNAP))
    return X, a_t


def w_mvdr(R, s, delta_frac=0.0):
    R1 = R + delta_frac * (np.trace(R).real / M) * np.eye(M)
    Rinv = np.linalg.inv(R1)
    return (Rinv @ s) / (s.conj().T @ Rinv @ s)


def w_patent(R, f, doa_deg, width_deg, c_slope=0.05, f_ref=2000.0):
    """LCMV over [s(δ-w/2), s(δ), s(δ+w/2)] with ε = c(slope)·b(f)·E(n), b(f) = (f/f_ref)^2."""
    A = np.hstack([steer(doa_deg - width_deg/2, f), steer(doa_deg, f), steer(doa_deg + width_deg/2, f)])
    Delta = np.ones((3, 1))
    E = np.trace(R).real / M                       # mic input power E(n)
    eps = c_slope * (f / f_ref)**2 * E             # ε(f, n, E, slope)
    R1 = R + eps * np.eye(M)
    R1inv = np.linalg.inv(R1)
    lam = np.linalg.inv(A.conj().T @ R1inv @ A) @ Delta
    return R1inv @ A @ lam


def snr_db(ref, est):
    return 10*np.log10(np.mean(np.abs(ref)**2) / np.mean(np.abs(est - ref)**2))


def pattern_db(w, f, thetas):
    S = np.hstack([steer(th, f) for th in thetas])
    return 20*np.log10(np.abs(w.conj().T @ S).squeeze() + 1e-12)


def ripple_db(pat, thetas, center, width):
    """Largest deviation from 0 dB inside [center - width/2, center + width/2]."""
    m = np.abs(thetas - center) <= width / 2
    return np.max(np.abs(pat[m]))


thetas = np.linspace(-90, 90, 3601)
METHODS = ["MVDR δ=0", "MVDR δ=0.1 (const)", "patent: ±2.5° boundary + ε(f)"]


def weights_for(R, f, doa):
    return [w_mvdr(R, steer(doa, f)), w_mvdr(R, steer(doa, f), 0.1), w_patent(R, f, doa, BEAM_WIDTH_DEG)]


# --- (a) patterns and (b) SNR vs steering error at F_SHOW
X, a_t = simulate(F_SHOW)
R = (X @ X.conj().T) / N_SNAP
W = weights_for(R, F_SHOW, DOA_EST_DEG)
pats = [pattern_db(w, F_SHOW, thetas) for w in W]
errors = np.arange(-6, 6.1, 0.5)
snr_err = {m: [] for m in METHODS}
for e in errors:
    for m, w in zip(METHODS, weights_for(R, F_SHOW, TARGET_DEG + e)):
        snr_err[m].append(snr_db(a_t, (w.conj().T @ X).squeeze()))

print(f"f = {F_SHOW} Hz, target {TARGET_DEG}°, steered to {DOA_EST_DEG}°, interferer {INTERF_DEG}° at +{INTERF_DB:.0f} dB")
print(f"  input SNR at mic 0: {snr_db(a_t, X[0]):.1f} dB")
for m, w, p in zip(METHODS, W, pats):
    g_t = p[np.argmin(np.abs(thetas - TARGET_DEG))]; g_i = p[np.argmin(np.abs(thetas - INTERF_DEG))]
    print(f"  {m:32s} gain@20° {g_t:+6.1f} dB   gain@−40° {g_i:+6.1f} dB   out SNR {snr_db(a_t, (w.conj().T @ X).squeeze()):5.1f} dB   "
          f"ripple over ±{BEAM_WIDTH_DEG/2:.1f}° {ripple_db(p, thetas, DOA_EST_DEG, BEAM_WIDTH_DEG):.1f} dB")

# --- (c) gain ripple across the beam width vs frequency, (d) SNR vs frequency (2 deg error)
bw = {m: [] for m in METHODS}
snr_f = {m: [] for m in METHODS}
for f in FREQS:
    Xf, af = simulate(f)
    Rf = (Xf @ Xf.conj().T) / N_SNAP
    for m, w in zip(METHODS, weights_for(Rf, f, DOA_EST_DEG)):
        bw[m].append(ripple_db(pattern_db(w, f, thetas), thetas, DOA_EST_DEG, BEAM_WIDTH_DEG))
        snr_f[m].append(snr_db(af, (w.conj().T @ Xf).squeeze()))
print(f"gain ripple across the ±{BEAM_WIDTH_DEG/2:.1f}° beam width vs frequency [dB]:")
for m in METHODS:
    print(f"  {m:32s}", np.round(bw[m], 1))

# --------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
cols = ["C3", "C1", "C2"]

ax = axes[0, 0]
for p, m, c in zip(pats, METHODS, cols):
    ax.plot(thetas, p, color=c, lw=1.3, label=m)
ax.axvspan(DOA_EST_DEG - BEAM_WIDTH_DEG/2, DOA_EST_DEG + BEAM_WIDTH_DEG/2, color="C2", alpha=0.12, label="pre-selected beam width")
ax.axvline(TARGET_DEG, color="k", ls=":", lw=1); ax.axvline(INTERF_DEG, color="r", ls=":", lw=1)
ax.text(TARGET_DEG - 12, -55, "true 20°", fontsize=8)
ax.set_xlim(-90, 90); ax.set_ylim(-60, 5); ax.set_xlabel("θ [deg]"); ax.set_ylabel("|wᴴs(θ)| [dB]")
ax.set_title(f"Beam patterns at {F_SHOW} Hz, steered to {DOA_EST_DEG:.0f}° (true target {TARGET_DEG:.0f}°)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)

ax = axes[0, 1]
for m, c in zip(METHODS, cols):
    ax.plot(errors, snr_err[m], color=c, lw=1.5, marker="o", ms=3, label=m)
ax.axvspan(-BEAM_WIDTH_DEG/2, BEAM_WIDTH_DEG/2, color="C2", alpha=0.12)
ax.set_xlabel("steering error [deg]"); ax.set_ylabel("output SNR [dB]")
ax.set_title("Output SNR vs DOA error (shaded = patent beam width)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)

ax = axes[1, 0]
for m, c in zip(METHODS, cols):
    ax.plot(FREQS, bw[m], color=c, lw=1.5, marker="s", ms=4, label=m)
ax.set_xlabel("frequency [Hz]"); ax.set_ylabel(f"max |gain| deviation over ±{BEAM_WIDTH_DEG/2:.1f}° [dB]")
ax.set_title("Gain ripple across the pre-selected beam width vs frequency\n(patent goal: uniform, frequency-independent)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)

ax = axes[1, 1]
for m, c in zip(METHODS, cols):
    ax.plot(FREQS, snr_f[m], color=c, lw=1.5, marker="s", ms=4, label=m)
ax.set_xlabel("frequency [Hz]"); ax.set_ylabel("output SNR [dB]")
ax.set_title("Output SNR vs frequency with 2° DOA error"); ax.grid(alpha=0.3); ax.legend(fontsize=8)

fig.suptitle("US 2018/0176679 A1: LCMV with boundary steering vectors + frequency-scaled loading", y=0.995)
fig.tight_layout()
os.makedirs(PLOTS_DIR, exist_ok=True)
out = os.path.join(PLOTS_DIR, "patent_beamwidth_lcmv.png")
fig.savefig(out, dpi=130)
print(f"wrote {out}")
