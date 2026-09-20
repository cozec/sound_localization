"""MVDR self-nulling under steering mismatch, and how diagonal loading fixes it.

8-element ULA, d = 0.5. Target tone from 20 deg at 0 dB, interferer tone
from -40 deg at +10 dB, white noise. MVDR is steered to 22 deg (2 deg DOA
error). With delta = 0 the beamformer sees the target as "strong power not
at the look direction" and nulls it (self-nulling); adding delta*I to R
caps the small eigenvalues, penalizes large weights, and stops the null
forming while the strong interferer still gets nulled.

    w = (R + delta I)^-1 s / (s^H (R + delta I)^-1 s)
    delta -> 0  : MVDR (sharp, brittle)
    delta -> inf: delay-and-sum s/Nr (dull, safe)

Panels: output SNR vs delta for several steering errors; beam patterns at
three delta values; weight norm (white-noise gain) vs delta; output SNR vs
input SNR showing MVDR getting *worse* with a louder target when mismatched.

Output: plots/pysdr_diagonal_loading.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")

np.random.seed(0)
sample_rate = 1e6
N = 10000
t = np.arange(N)/sample_rate
d = 0.5
Nr = 8
TARGET_DEG, INTERF_DEG = 20.0, -40.0
INTERF_AMP = np.sqrt(10)           # +10 dB
NOISE = 0.3


def steer(theta_deg):
    return np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(np.radians(theta_deg))).reshape(-1, 1)


def simulate(target_amp=1.0):
    tx = np.exp(2j*np.pi*0.01e6*t)
    X = target_amp * steer(TARGET_DEG) @ tx.reshape(1, -1)
    X += INTERF_AMP * steer(INTERF_DEG) @ np.exp(2j*np.pi*0.03e6*t).reshape(1, -1)
    X += NOISE * (np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N))
    return X, target_amp * tx


def w_mvdr_loaded(theta_deg, R, delta_frac):
    """delta expressed as a fraction of the average channel power trace(R)/Nr."""
    s = steer(theta_deg)
    Rl = R + delta_frac * (np.trace(R).real / Nr) * np.eye(Nr)
    Rinv = np.linalg.inv(Rl)
    return (Rinv @ s) / (s.conj().T @ Rinv @ s)


def snr_db(ref, est):
    return 10*np.log10(np.mean(np.abs(ref)**2) / np.mean(np.abs(est - ref)**2))


X, tx = simulate()
R = (X @ X.conj().T) / N
deltas = np.logspace(-4, 2, 61)
errors = [0.0, 1.0, 2.0, 4.0]

# --- (1) output SNR vs delta for several steering errors
snr_curves = {}
for err in errors:
    snr_curves[err] = [snr_db(tx, (w_mvdr_loaded(TARGET_DEG + err, R, dl).conj().T @ X).squeeze()) for dl in deltas]
w_das = steer(TARGET_DEG + 2.0) / Nr
snr_das = snr_db(tx, (w_das.conj().T @ X).squeeze())
snr_in = snr_db(tx, X[0])

print(f"input SNR at element 0: {snr_in:.1f} dB   (interferer +10 dB, noise {NOISE})")
print(f"delay-and-sum steered to 22°: {snr_das:.1f} dB")
for err in errors:
    c = np.array(snr_curves[err])
    print(f"steer error {err:+.0f}°:  MVDR δ→0 {c[0]:6.1f} dB   best {c.max():6.1f} dB at δ={deltas[c.argmax()]:.3g}   δ=1 {c[np.argmin(abs(deltas-1))]:6.1f} dB")

# --- (2) beam patterns at 2 deg error, three deltas
theta_pat = np.linspace(-90, 90, 1441)
S = np.hstack([steer(th) for th in theta_pat])
pats = {}
for dl in (1e-4, 1e-1, 1e1):
    w = w_mvdr_loaded(TARGET_DEG + 2.0, R, dl)
    pats[dl] = 20*np.log10(np.abs(w.conj().T @ S).squeeze() + 1e-12)

# --- (3) weight norm (white-noise gain = 1/||w||^2) vs delta
wng = [10*np.log10(1/np.linalg.norm(w_mvdr_loaded(TARGET_DEG + 2.0, R, dl))**2) for dl in deltas]

# --- (4) output SNR vs target level, 2 deg error, delta = 0 vs loaded
amps_db = np.arange(-10, 31, 5)
snr_vs_level = {0: [], 1e-1: [], "das": []}
for a_db in amps_db:
    Xa, txa = simulate(10**(a_db/20))
    Ra = (Xa @ Xa.conj().T) / N
    for dl in (0, 1e-1):
        snr_vs_level[dl].append(snr_db(txa, (w_mvdr_loaded(TARGET_DEG + 2.0, Ra, dl).conj().T @ Xa).squeeze()))
    snr_vs_level["das"].append(snr_db(txa, (w_das.conj().T @ Xa).squeeze()))

# --------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

ax = axes[0, 0]
for err in errors:
    ax.semilogx(deltas, snr_curves[err], lw=1.5, label=f"steer error {err:+.0f}°")
ax.axhline(snr_das, color="k", ls="--", lw=1, label="delay-and-sum (22°)")
ax.axhline(snr_in, color="0.5", ls=":", lw=1, label="input (element 0)")
ax.set_xlabel("δ  [fraction of trace(R)/Nr]"); ax.set_ylabel("output SNR [dB]")
ax.set_title("Output SNR vs diagonal loading — target 20°, interferer −40° (+10 dB)")
ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)

ax = axes[0, 1]
for (dl, p), c in zip(pats.items(), ("C3", "C2", "C0")):
    ax.plot(theta_pat, p, color=c, lw=1.3, label=f"δ = {dl:g}")
ax.axvline(TARGET_DEG, color="k", ls=":", lw=1); ax.axvline(TARGET_DEG + 2, color="0.4", ls="--", lw=0.8)
ax.axvline(INTERF_DEG, color="r", ls=":", lw=1)
ax.text(TARGET_DEG + 2.5, 2, "look 22°", fontsize=8); ax.text(TARGET_DEG - 9, -55, "true 20°", fontsize=8)
ax.set_xlim(-90, 90); ax.set_ylim(-60, 5); ax.set_xlabel("θ [deg]"); ax.set_ylabel("|wᴴs(θ)| [dB]")
ax.set_title("Beam pattern steered to 22° (true target 20°)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)

ax = axes[1, 0]
ax.semilogx(deltas, wng, lw=1.5)
ax.axhline(10*np.log10(Nr), color="k", ls="--", lw=1, label=f"delay-and-sum limit 10·log10(Nr) = {10*np.log10(Nr):.1f} dB")
ax.set_xlabel("δ"); ax.set_ylabel("white-noise gain 1/‖w‖² [dB]")
ax.set_title("Loading bounds the weight norm (2° error)"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)

ax = axes[1, 1]
ax.plot(amps_db, snr_vs_level[0], "C3o-", lw=1.5, label="MVDR δ = 0")
ax.plot(amps_db, snr_vs_level[1e-1], "C2s-", lw=1.5, label="MVDR δ = 0.1")
ax.plot(amps_db, snr_vs_level["das"], "k^--", lw=1, label="delay-and-sum")
ax.set_xlabel("target level [dB re. 0 dB case]"); ax.set_ylabel("output SNR [dB]")
ax.set_title("Self-nulling gets worse as the target gets louder (2° error)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)

fig.tight_layout()
os.makedirs(PLOTS_DIR, exist_ok=True)
out = os.path.join(PLOTS_DIR, "pysdr_diagonal_loading.png")
fig.savefig(out, dpi=130)
print(f"wrote {out}")
