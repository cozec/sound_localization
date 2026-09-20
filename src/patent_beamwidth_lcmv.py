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
# Figure: one idea per panel
# --------------------------------------------------------------------------
STYLE = {METHODS[0]: dict(color="#c0392b", lw=1.6, ls="-",  label="MVDR, no loading"),
         METHODS[1]: dict(color="#e67e22", lw=1.2, ls="--", label="MVDR, δ = 0.1"),
         METHODS[2]: dict(color="#1e8449", lw=2.4, ls="-",  label="patent: 3 constraints ±2.5° + ε(f)")}
cons_deg = [DOA_EST_DEG - BEAM_WIDTH_DEG/2, DOA_EST_DEG, DOA_EST_DEG + BEAM_WIDTH_DEG/2]

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.patch.set_facecolor("white")

# (a) zoom on the beam: where the constraints sit and where the target is
ax = axes[0, 0]
for m, p in zip(METHODS, pats):
    ax.plot(thetas, p, **STYLE[m])
ax.axvspan(cons_deg[0], cons_deg[2], color="#1e8449", alpha=0.08)
ax.plot(cons_deg, [0, 0, 0], "o", ms=9, mfc="white", mec="#1e8449", mew=2, zorder=5,
        label="constraints: Cᴴw = [1,1,1]")
ax.axvline(TARGET_DEG, color="k", ls=":", lw=1.2)
ax.annotate("true target 20°", xy=(TARGET_DEG, -13), xytext=(8, -24), fontsize=10,
            arrowprops=dict(arrowstyle="->", lw=1))
ax.annotate("MVDR self-null\n−14 dB", xy=(TARGET_DEG, pats[0][np.argmin(np.abs(thetas - TARGET_DEG))]),
            xytext=(24.5, -20), fontsize=10, color="#c0392b", arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1))
ax.annotate("patent: 0 dB across\nthe whole beam width", xy=(cons_deg[2], 0.3), xytext=(27.5, -8), fontsize=10,
            color="#1e8449", arrowprops=dict(arrowstyle="->", color="#1e8449", lw=1))
ax.set_xlim(5, 40); ax.set_ylim(-40, 10)
ax.set_xlabel("θ [deg]"); ax.set_ylabel("gain |wᴴs(θ)| [dB]")
ax.set_title(f"(a) Beam around the look direction, {F_SHOW} Hz — steered to {DOA_EST_DEG:.0f}°", loc="left", fontsize=12)
ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=9)

# (b) full pattern, MVDR vs patent only
ax = axes[0, 1]
for m in (METHODS[0], METHODS[2]):
    ax.plot(thetas, pats[METHODS.index(m)], **STYLE[m])
ax.axvspan(cons_deg[0], cons_deg[2], color="#1e8449", alpha=0.12)
ax.axvline(INTERF_DEG, color="k", ls=":", lw=1.2)
ax.annotate("interferer −40° (+10 dB)\nnulled by both", xy=(INTERF_DEG, -45), xytext=(-30, -50), fontsize=10,
            arrowprops=dict(arrowstyle="->", lw=1))
ax.annotate("price: sidelobes next to\nthe flat top rise to +4 dB", xy=(30, 4), xytext=(45, -12), fontsize=10,
            color="#1e8449", arrowprops=dict(arrowstyle="->", color="#1e8449", lw=1))
ax.set_xlim(-90, 90); ax.set_ylim(-60, 8)
ax.set_xlabel("θ [deg]"); ax.set_ylabel("gain [dB]")
ax.set_title(f"(b) Full pattern, {F_SHOW} Hz", loc="left", fontsize=12)
ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=9)

# (c) SNR vs DOA error
ax = axes[1, 0]
ax.axvspan(-BEAM_WIDTH_DEG/2, BEAM_WIDTH_DEG/2, color="#1e8449", alpha=0.08)
for m in METHODS:
    ax.plot(errors, snr_err[m], marker="o", ms=3.5, **STYLE[m])
ax.text(0, 12.3, "pre-selected beam width (5°)", ha="center", fontsize=9, color="#1e8449")
ax.set_xlabel("DOA estimation error [deg]"); ax.set_ylabel("output SNR [dB]")
ax.set_title(f"(c) Robustness to DOA error, {F_SHOW} Hz", loc="left", fontsize=12)
ax.set_ylim(-1, 21); ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=9)

# (d) SNR vs frequency with 2° error
ax = axes[1, 1]
for m in METHODS:
    ax.plot(FREQS, snr_f[m], marker="s", ms=5, **STYLE[m])
ax.set_xlabel("frequency [Hz]"); ax.set_ylabel("output SNR [dB]")
ax.set_title("(d) Same 2° DOA error, across frequency", loc="left", fontsize=12)
ax.set_ylim(-1, 18); ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=9)
ax.text(2500, 12.4, "flat = 'standardized regardless of frequency'", fontsize=9, color="#1e8449", ha="center")

fig.suptitle("US 2018/0176679 A1 — hold unit gain across a pre-selected beam width so a DOA error cannot self-null the target\n"
             f"{M} mics, {D_M*100:.0f} cm spacing · target 20° at 0 dB · DOA estimate 22° · interferer −40° at +10 dB",
             fontsize=12, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.96))
os.makedirs(PLOTS_DIR, exist_ok=True)
out = os.path.join(PLOTS_DIR, "patent_beamwidth_lcmv.png")
fig.savefig(out, dpi=130)
print(f"wrote {out}")
