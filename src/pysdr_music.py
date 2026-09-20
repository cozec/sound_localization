"""PySDR "MUSIC" section as a standalone script.

Code from https://pysdr.org/content/doa.html (Marc Lichtman, PySDR,
CC BY-NC-SA 4.0). Comments are the original author's. Uses the same
8-element / three-source scenario as the MVDR section (20, 25 and -40 deg,
the last at 1/10 amplitude).

MUSIC (MUltiple SIgnal Classification) is a subspace method: eigendecompose
R, call the Nr - K smallest-eigenvalue eigenvectors the "noise subspace" V,
and score each angle by how *orthogonal* its steering vector is to V:
    P(theta) = 1 / (s^H V V^H s)
A true source direction lies in the signal subspace, so s^H V ~ 0 and the
metric spikes. Needs K (num_expected_signals) up front; the eigenvalue plot
is how you pick it.

Panels: eigenvalues of R; MUSIC vs MVDR vs conventional scans; MUSIC with
K = 1, 3, 5 (the page's "Try changing this!"); Root MUSIC estimates.

Output: plots/pysdr_music.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")

np.random.seed(0)

sample_rate = 1e6
N = 10000 # number of samples to simulate
t = np.arange(N)/sample_rate # time vector
d = 0.5 # half wavelength spacing

# Scenario (from the MVDR section)
Nr = 8 # 8 elements
theta1 = 20 / 180 * np.pi # convert to radians
theta2 = 25 / 180 * np.pi
theta3 = -40 / 180 * np.pi
s1 = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta1)).reshape(-1,1) # 8x1
s2 = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta2)).reshape(-1,1)
s3 = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta3)).reshape(-1,1)
# we'll use 3 different frequencies.  1xN
tone1 = np.exp(2j*np.pi*0.01e6*t).reshape(1,-1)
tone2 = np.exp(2j*np.pi*0.02e6*t).reshape(1,-1)
tone3 = np.exp(2j*np.pi*0.03e6*t).reshape(1,-1)
X = s1 @ tone1 + s2 @ tone2 + 0.1 * s3 @ tone3 # note the last one is 1/10th the power
n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
X = X + 0.05*n # 8xN

TRUE_DEG = (20, 25, -40)
theta_scan = np.linspace(-1*np.pi/2, np.pi/2, 2000) # -90 to +90 degrees (the page scans +-180; a ULA is mirror-symmetric)


def music_scan(X, num_expected_signals):
    """MUSIC pseudo-spectrum in dB, normalized to 0 dB at the peak (page code, wrapped)."""
    # part that doesn't change with theta_i
    R = np.cov(X) # Calc covariance matrix. gives a Nr x Nr covariance matrix
    w, v = np.linalg.eig(R) # eigenvalue decomposition, v[:,i] is the eigenvector corresponding to the eigenvalue w[i]
    eig_val_order = np.argsort(np.abs(w)) # find order of magnitude of eigenvalues
    v = v[:, eig_val_order] # sort eigenvectors using this order
    # We make a new eigenvector matrix representing the "noise subspace", it's just the rest of the eigenvalues
    V = np.zeros((Nr, Nr - num_expected_signals), dtype=np.complex64)
    for i in range(Nr - num_expected_signals):
       V[:, i] = v[:, i]

    results = []
    for theta_i in theta_scan:
        s = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta_i)) # Steering Vector
        s = s.reshape(-1,1)
        metric = 1 / (s.conj().T @ V @ V.conj().T @ s) # The main MUSIC equation
        metric = np.abs(metric.squeeze()) # take magnitude
        metric = 10*np.log10(metric) # convert to dB
        results.append(metric)

    results -= np.max(results) # normalize (page uses /=, which is wrong for dB)
    return np.array(results), np.abs(w[eig_val_order])


def mvdr_scan(X):
    R = (X @ X.conj().T)/X.shape[1]
    Rinv = np.linalg.pinv(R)
    out = []
    for theta_i in theta_scan:
        s = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta_i)).reshape(-1,1)
        out.append(10*np.log10(np.abs(1/(s.conj().T @ Rinv @ s).squeeze())))
    return np.array(out) - np.max(out)


def conventional_scan(X):
    out = []
    for theta_i in theta_scan:
        w = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta_i))
        out.append(10*np.log10(np.var(w.conj().T @ X)))
    return np.array(out) - np.max(out)


def root_music(X, num_expected_signals):
    """Closed-form DOAs from the Root MUSIC section."""
    R = np.cov(X)
    w, v = np.linalg.eig(R)
    eig_val_order = np.argsort(np.abs(w))
    v = v[:, eig_val_order]
    V = v[:, :Nr - num_expected_signals]  # noise subspace eigenvectors
    D = V @ V.conj().T
    p = np.zeros(2*Nr - 1, dtype=np.complex128)
    for k in range(2*Nr - 1):
        p[k] = np.sum(np.diag(D, k - (Nr - 1)))
    roots = np.roots(p[::-1])
    roots = roots[np.abs(roots) <= 1.0]
    roots = roots[np.argsort(-np.abs(roots))]
    doa_roots = roots[:num_expected_signals]
    return np.sort(np.arcsin(np.angle(doa_roots) / (2 * np.pi * d)) * 180 / np.pi)


music3, eigvals = music_scan(X, 3)
mvdr = mvdr_scan(X)
conv = conventional_scan(X)
deg = np.degrees(theta_scan)

np.set_printoptions(precision=3, suppress=True)
print("eigenvalues of R (ascending):", eigvals)
print("  -> 5 small (noise, ~", np.round(eigvals[:5].mean(), 4), ") and 3 large (signal): pick num_expected_signals = 3")
from scipy.signal import find_peaks
for name, sc in (("conventional", conv), ("MVDR", mvdr), ("MUSIC K=3", music3)):
    pk, pr = find_peaks(sc, prominence=3)
    print(f"{name:13s} peaks: {[round(deg[i], 1) for i in pk]}")
for K in (1, 2, 3, 4, 5):
    print(f"Root MUSIC K={K}: {root_music(X, K)}")

# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

ax = axes[0, 0]
ax.plot(10*np.log10(np.abs(eigvals)), '.-', markersize=10)   # the page's `plot(10*np.log10(np.abs(w)),'.-')`
ax.axvline(Nr - 3 - 0.5, color="k", ls=":", lw=0.8)
ax.text(Nr - 3 - 0.4, ax.get_ylim()[1] - 3, "noise | signal", fontsize=9)
ax.set_xlabel("eigenvalue index (sorted ascending)"); ax.set_ylabel("dB")
ax.set_title("Eigenvalues of R: 5 noise + 3 signal → K = 3"); ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.plot(deg, conv, label="conventional", lw=1)
ax.plot(deg, mvdr, label="MVDR", lw=1)
ax.plot(deg, music3, label="MUSIC (K=3)", lw=1.5)
for a in TRUE_DEG:
    ax.axvline(a, color="k", lw=0.8, ls=":")
ax.set_xlim(-90, 90); ax.set_xlabel("Theta [deg]"); ax.set_ylabel("dB")
ax.set_title("Three scans, sources at 20°, 25°, −40° (−20 dB)"); ax.grid(); ax.legend()

ax = axes[1, 0]
for K, c in zip((1, 3, 5), ("C3", "C2", "C4")):
    sc, _ = music_scan(X, K)
    ax.plot(deg, sc, color=c, lw=1.2, label=f"MUSIC K={K}")
for a in TRUE_DEG:
    ax.axvline(a, color="k", lw=0.8, ls=":")
ax.set_xlim(-90, 90); ax.set_xlabel("Theta [deg]"); ax.set_ylabel("dB")
ax.set_title("Effect of num_expected_signals (true K = 3)"); ax.grid(); ax.legend()

ax = axes[1, 1]
ax.plot(deg, mvdr, label="MVDR", lw=1)
ax.plot(deg, music3, label="MUSIC (K=3)", lw=1.5)
for a in TRUE_DEG[:2]:
    ax.axvline(a, color="k", lw=0.8, ls=":")
ax.set_xlim(15, 30); ax.set_ylim(-40, 1)
ax.set_xlabel("Theta [deg]"); ax.set_title(f"Zoom 20°/25° — Root MUSIC: {np.round(root_music(X, 3), 2)}")
ax.grid(); ax.legend()

fig.tight_layout()
os.makedirs(PLOTS_DIR, exist_ok=True)
out = os.path.join(PLOTS_DIR, "pysdr_music.png")
fig.savefig(out, dpi=130)
print(f"wrote {out}")
