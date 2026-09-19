"""Part B of pysdr_mvdr.py with waveforms, weights and beam patterns.

8-element ULA, d = 0.5. Tones from +20 deg (target), +25 deg (equal-power
interferer, 5 deg away) and -40 deg (1/10 amplitude), white noise 0.05.
Both beamformers are steered to +20 deg. Shows inputs, outputs vs the clean
target tone, the DOA scans, and the beam pattern |w^H s(theta)| with MVDR
nulls on both interferers. Weights and gains are printed.

Output: plots/pysdr_mvdr_part_b.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True, linewidth=120)

sample_rate = 1e6
N = 10000
t = np.arange(N)/sample_rate
d = 0.5
Nr = 8

SOURCES = [(20, 0.01e6, 1.0), (25, 0.02e6, 1.0), (-40, 0.03e6, 0.1)]  # (deg, tone Hz, amplitude)
TARGET_DEG = SOURCES[0][0]
theta_t = TARGET_DEG / 180 * np.pi


def steer(theta):
    return np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta)).reshape(-1, 1)


X = np.zeros((Nr, N), dtype=complex)
for deg_, f_, amp_ in SOURCES:
    X += amp_ * steer(np.radians(deg_)) @ np.exp(2j*np.pi*f_*t).reshape(1, -1)
n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
X = X + 0.05*n
tx = np.exp(2j*np.pi*SOURCES[0][1]*t)   # clean target tone


def w_mvdr(theta, X):
    s = steer(theta)
    R = (X @ X.conj().T)/X.shape[1]
    Rinv = np.linalg.pinv(R)
    return (Rinv @ s)/(s.conj().T @ Rinv @ s)


w_conv = steer(theta_t) / Nr
w_m = w_mvdr(theta_t, X)
y_conv = (w_conv.conj().T @ X).squeeze()
y_mvdr = (w_m.conj().T @ X).squeeze()

print("w_conv =", w_conv.squeeze())
print("w_mvdr =", w_m.squeeze())
print("|w_conv| =", np.abs(w_conv.squeeze()))
print("|w_mvdr| =", np.abs(w_m.squeeze()))
print("phase w_conv deg =", np.degrees(np.angle(w_conv.squeeze())))
print("phase w_mvdr deg =", np.degrees(np.angle(w_m.squeeze())))
for name, w in (("conv", w_conv), ("mvdr", w_m)):
    gains = [abs((w.conj().T @ steer(np.radians(deg_))).squeeze()) for deg_, _, _ in SOURCES]
    print(f"{name}: " + "  ".join(f"gain@{deg_:+d}° {g:.4f} ({20*np.log10(g):+.1f} dB)"
                                  for (deg_, _, _), g in zip(SOURCES, gains)))
for name, y in (("element 0", X[0]), ("conventional", y_conv), ("MVDR", y_mvdr)):
    err = y - tx
    print("SNR %-12s %6.2f dB" % (name, 10*np.log10(np.mean(np.abs(tx)**2)/np.mean(np.abs(err)**2))))

theta_scan = np.linspace(-np.pi/2, np.pi/2, 2000)
S = np.hstack([steer(th) for th in theta_scan])
conv_scan = np.array([10*np.log10(np.var((steer(th)/Nr).conj().T @ X)) for th in theta_scan])
R = (X @ X.conj().T)/X.shape[1]; Rinv = np.linalg.pinv(R)
mvdr_scan = np.array([10*np.log10(abs(1/(steer(th).conj().T @ Rinv @ steer(th)).squeeze())) for th in theta_scan])
conv_scan -= conv_scan.max(); mvdr_scan -= mvdr_scan.max()
pat_conv = 20*np.log10(np.abs(w_conv.conj().T @ S).squeeze() + 1e-12)
pat_mvdr = 20*np.log10(np.abs(w_m.conj().T @ S).squeeze() + 1e-12)

deg = np.degrees(theta_scan)
M = 400
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

ax = axes[0, 0]
for k in range(Nr):
    ax.plot(X[k, :M].real, lw=0.7, label=f"el {k}")
ax.set_title("Inputs: +20° (target) + 25° (equal) + −40° (−20 dB) + noise")
ax.legend(loc="upper right", ncol=4, fontsize=8); ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.plot(tx[:M].real, "k", lw=1, label="clean target tone (10 kHz)")
ax.plot(y_conv[:M].real, "C0", lw=1, label="conventional output")
ax.plot(y_mvdr[:M].real, "C1", lw=1, linestyle="--", label="MVDR output")
ax.set_title(f"Outputs steered to {TARGET_DEG:+d}°"); ax.legend(loc="upper right"); ax.grid(alpha=0.3)
ax.set_xlabel("sample")

ax = axes[1, 0]
ax.plot(deg, conv_scan, label="conventional"); ax.plot(deg, mvdr_scan, label="MVDR")
for deg_, _, _ in SOURCES:
    ax.axvline(deg_, color="k", lw=0.8, ls=":")
ax.set_title("DOA scan"); ax.set_xlabel("Theta [deg]"); ax.set_ylabel("dB")
ax.set_xlim(-90, 90); ax.grid(); ax.legend()

ax = axes[1, 1]
ax.plot(deg, pat_conv, label="conventional |wᴴs(θ)|"); ax.plot(deg, pat_mvdr, label="MVDR |wᴴs(θ)|")
ax.axvline(TARGET_DEG, color="k", lw=0.8, ls=":")
for deg_, _, _ in SOURCES[1:]:
    ax.axvline(deg_, color="r", lw=0.8, ls=":")
ax.set_title(f"Beam pattern steered to {TARGET_DEG:+d}° (red = interferers at +25°, −40°)")
ax.set_xlabel("Theta [deg]"); ax.set_ylabel("dB"); ax.set_xlim(-90, 90); ax.set_ylim(-60, 5); ax.grid(); ax.legend()

fig.tight_layout()
os.makedirs(PLOTS_DIR, exist_ok=True)
out = os.path.join(PLOTS_DIR, "pysdr_mvdr_part_b.png")
fig.savefig(out, dpi=130)
print(f"wrote {out}")
