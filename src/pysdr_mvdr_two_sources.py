"""Part A of pysdr_mvdr.py with a second (interfering) source.

3-element ULA, d = 0.5. Target tone from +20 deg, interferer tone (different
frequency, same amplitude) from -40 deg, white noise 0.1. Both beamformers
are steered to +20 deg; the plot shows the inputs, the two outputs against
the clean target tone, the DOA scans, and the beam patterns |w^H s(theta)|
so the MVDR null on the interferer is visible. Weights are printed.

Output: plots/pysdr_mvdr_two_sources.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")

np.random.seed(0)
np.set_printoptions(precision=4, suppress=True)

sample_rate = 1e6
N = 10000
t = np.arange(N)/sample_rate
d = 0.5
Nr = 3

TARGET_DEG = 20
INTERF_DEG = -40
theta_t = TARGET_DEG / 180 * np.pi
theta_i = INTERF_DEG / 180 * np.pi


def steer(theta):
    return np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta)).reshape(-1, 1)


tone_t = np.exp(2j*np.pi*0.02e6*t).reshape(1, -1)   # target
tone_i = np.exp(2j*np.pi*0.03e6*t).reshape(1, -1)   # interferer, same amplitude
X = steer(theta_t) @ tone_t + steer(theta_i) @ tone_i
n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
X = X + 0.1*n


def w_mvdr(theta, X):
    s = steer(theta)
    R = (X @ X.conj().T)/X.shape[1]
    Rinv = np.linalg.pinv(R)
    return (Rinv @ s)/(s.conj().T @ Rinv @ s)


w_conv = steer(theta_t) / Nr
w_m = w_mvdr(theta_t, X)
y_conv = (w_conv.conj().T @ X).squeeze()
y_mvdr = (w_m.conj().T @ X).squeeze()
tx = tone_t.squeeze()

print("w_conv =", w_conv.squeeze())
print("w_mvdr =", w_m.squeeze())
print("|w_conv| =", np.abs(w_conv.squeeze()), " phase deg =", np.degrees(np.angle(w_conv.squeeze())))
print("|w_mvdr| =", np.abs(w_m.squeeze()), " phase deg =", np.degrees(np.angle(w_m.squeeze())))
for name, w in (("conv", w_conv), ("mvdr", w_m)):
    g_t = (w.conj().T @ steer(theta_t)).squeeze()
    g_i = (w.conj().T @ steer(theta_i)).squeeze()
    print(f"{name}: gain at target {abs(g_t):.4f} ({20*np.log10(abs(g_t)):+.1f} dB), "
          f"at interferer {abs(g_i):.4f} ({20*np.log10(abs(g_i)):+.1f} dB)")
for name, y in (("element 0", X[0]), ("conventional", y_conv), ("MVDR", y_mvdr)):
    err = y - tx
    print("SNR %-12s %6.2f dB" % (name, 10*np.log10(np.mean(np.abs(tx)**2)/np.mean(np.abs(err)**2))))

# Scans and beam patterns
theta_scan = np.linspace(-np.pi/2, np.pi/2, 1000)
S = np.hstack([steer(th) for th in theta_scan])              # Nr x n_angles
conv_scan = np.array([10*np.log10(np.var((steer(th)/Nr).conj().T @ X)) for th in theta_scan])
mvdr_scan = np.array([10*np.log10(np.var(w_mvdr(th, X).conj().T @ X)) for th in theta_scan])
conv_scan -= conv_scan.max(); mvdr_scan -= mvdr_scan.max()
pat_conv = 20*np.log10(np.abs(w_conv.conj().T @ S).squeeze() + 1e-12)
pat_mvdr = 20*np.log10(np.abs(w_m.conj().T @ S).squeeze() + 1e-12)

deg = np.degrees(theta_scan)
M = 200
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

ax = axes[0, 0]
for k in range(Nr):
    ax.plot(X[k, :M].real, lw=0.8, label=f"element {k}")
ax.set_title(f"Inputs: target tone from {TARGET_DEG:+d}° + interferer tone from {INTERF_DEG:+d}° + noise")
ax.legend(loc="upper right"); ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.plot(tx[:M].real, "k", lw=1, label="clean target tone")
ax.plot(y_conv[:M].real, "C0", lw=1, label="conventional output")
ax.plot(y_mvdr[:M].real, "C1", lw=1, linestyle="--", label="MVDR output")
ax.set_title(f"Outputs steered to {TARGET_DEG:+d}°"); ax.legend(loc="upper right"); ax.grid(alpha=0.3)
ax.set_xlabel("sample")

ax = axes[1, 0]
ax.plot(deg, conv_scan, label="conventional"); ax.plot(deg, mvdr_scan, label="MVDR")
for a in (TARGET_DEG, INTERF_DEG):
    ax.axvline(a, color="k", lw=0.8, ls=":")
ax.set_title("DOA scan (output power vs look direction)"); ax.set_xlabel("Theta [deg]"); ax.set_ylabel("dB")
ax.set_xlim(-90, 90); ax.grid(); ax.legend()

ax = axes[1, 1]
ax.plot(deg, pat_conv, label="conventional |wᴴs(θ)|"); ax.plot(deg, pat_mvdr, label="MVDR |wᴴs(θ)|")
ax.axvline(TARGET_DEG, color="k", lw=0.8, ls=":"); ax.axvline(INTERF_DEG, color="r", lw=0.8, ls=":")
ax.set_title(f"Beam pattern of the weights steered to {TARGET_DEG:+d}° (red = interferer)")
ax.set_xlabel("Theta [deg]"); ax.set_ylabel("dB"); ax.set_xlim(-90, 90); ax.set_ylim(-50, 5); ax.grid(); ax.legend()

fig.tight_layout()
os.makedirs(PLOTS_DIR, exist_ok=True)
out = os.path.join(PLOTS_DIR, "pysdr_mvdr_two_sources.png")
fig.savefig(out, dpi=130)
print(f"wrote {out}")
