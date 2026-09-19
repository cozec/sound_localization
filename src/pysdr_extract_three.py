"""Extract all three sources of Part B once their DOAs are known.

Same scenario as pysdr_mvdr_part_b.py (8 elements, tones from +20, +25 and
-40 deg, the last at 1/10 amplitude, noise 0.05). Three ways to get one
output stream per source:

  1. MVDR bank      w_k = R^-1 s_k / (s_k^H R^-1 s_k)   -- one beam per DOA,
                    data-adaptive, no knowledge of the other DOAs needed.
  2. LCMV           W  = R^-1 C (C^H R^-1 C)^-1          -- C = [s1 s2 s3];
                    column k has unit gain at DOA k and exact zeros at the
                    other two (multi-constraint MVDR).
  3. Zero-forcing   W  = pinv(C)^H = C (C^H C)^-1        -- pure geometry, no
                    covariance; least-squares unmixing of X = C S + n.

Output: plots/pysdr_extract_three.png
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
SOURCES = [(20, 0.01e6, 1.0), (25, 0.02e6, 1.0), (-40, 0.03e6, 0.1)]  # (deg, tone Hz, amplitude)


def steer(theta_deg):
    return np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(np.radians(theta_deg))).reshape(-1, 1)


clean = np.vstack([amp * np.exp(2j*np.pi*f*t) for _, f, amp in SOURCES])   # 3 x N, as received at element 0
C = np.hstack([steer(deg) for deg, _, _ in SOURCES])                       # Nr x 3
X = C @ clean + 0.05*(np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N))

R = (X @ X.conj().T)/N
Rinv = np.linalg.pinv(R)

# 1. MVDR bank: Nr x 3, one column per DOA
W_mvdr = np.hstack([(Rinv @ steer(deg)) / (steer(deg).conj().T @ Rinv @ steer(deg)) for deg, _, _ in SOURCES])
# 2. LCMV with C^H W = I
W_lcmv = Rinv @ C @ np.linalg.inv(C.conj().T @ Rinv @ C)
# 3. Zero-forcing / least squares
W_zf = C @ np.linalg.inv(C.conj().T @ C)

methods = {"MVDR bank": W_mvdr, "LCMV": W_lcmv, "Zero-forcing (pinv)": W_zf}


def snr_db(ref, est):
    return 10*np.log10(np.mean(np.abs(ref)**2) / np.mean(np.abs(est - ref)**2))


print("Gain matrix W^H C (rows = output k, cols = source):")
for name, W in methods.items():
    Y = W.conj().T @ X                        # 3 x N
    G = np.abs(W.conj().T @ C)
    print(f"\n{name}\n{np.round(G, 4)}")
    print("  output SNR per source [dB]:",
          [round(snr_db(clean[k], Y[k]), 1) for k in range(3)],
          "  input SNR at element 0:",
          [round(snr_db(clean[k], X[0]), 1) for k in range(3)])

M = 300
fig, axes = plt.subplots(3, 3, figsize=(15, 9), sharex=True)
for j, (name, W) in enumerate(methods.items()):
    Y = W.conj().T @ X
    for k, (deg, f, amp) in enumerate(SOURCES):
        ax = axes[k, j]
        ax.plot(clean[k, :M].real, "k", lw=1, label="clean")
        ax.plot(Y[k, :M].real, f"C{k}", lw=1, ls="--", label="extracted")
        ax.set_title(f"{name}: source {k+1} @ {deg:+d}°, {f/1e3:.0f} kHz  (SNR {snr_db(clean[k], Y[k]):.1f} dB)",
                     fontsize=10)
        ax.grid(alpha=0.3)
        if k == 2:
            ax.set_xlabel("sample")
        if j == 0 and k == 0:
            ax.legend(loc="upper right")
fig.suptitle("Extracting three sources from 8 elements given their DOAs", y=0.995)
fig.tight_layout()
os.makedirs(PLOTS_DIR, exist_ok=True)
out = os.path.join(PLOTS_DIR, "pysdr_extract_three.png")
fig.savefig(out, dpi=130)
print(f"\nwrote {out}")
