"""PySDR "MVDR/Capon Beamformer" section as a standalone script.

Code from https://pysdr.org/content/doa.html (Marc Lichtman, PySDR,
CC BY-NC-SA 4.0). The section's four snippets are joined with the
simulation setup they depend on (sample_rate, t, N, d from earlier in
the chapter). Comments are the original author's.

Part A: 3-element ULA, single tone at 20 deg, MVDR scan via w_mvdr()
        (weights applied to X, output power measured) vs conventional.
Part B: 8-element ULA, tones at 20, 25 and -40 deg (last one at 1/10
        amplitude), MVDR scan via the closed form power_mvdr() = 1/(s^H R^-1 s)
        vs conventional -- shows MVDR resolving 20/25 deg where the
        conventional beam cannot.

Output: plots/pysdr_mvdr.png (scans), plots/pysdr_mvdr_waveforms.png (Part A waveforms)
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")

np.random.seed(0)

# Simulation setup (from "Receiving a Signal")
sample_rate = 1e6
N = 10000 # number of samples to simulate
t = np.arange(N)/sample_rate # time vector
d = 0.5 # half wavelength spacing


def conventional_scan(theta_scan, X):
    """Delay-and-sum scan (from "Conventional Beamforming & DOA"), for reference."""
    Nr = X.shape[0]
    results = []
    for theta_i in theta_scan:
        w = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta_i)) # Conventional, aka delay-and-sum, beamformer
        X_weighted = w.conj().T @ X # apply our weights
        results.append(10*np.log10(np.var(X_weighted)))
    results -= np.max(results)
    return results


# --------------------------------------------------------------------------
# Part A: Nr = 3, single source at 20 deg
# --------------------------------------------------------------------------
Nr = 3
theta_degrees = 20 # direction of arrival
theta = theta_degrees / 180 * np.pi
s = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta)).reshape(-1,1)
tx = np.exp(2j * np.pi * 0.02e6 * t).reshape(1,-1)
X = s @ tx
n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
X = X + 0.1*n # X and n are both 3x10000


# theta is the direction of interest, in radians, and X is our received signal
def w_mvdr(theta, X):
   s = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta)) # steering vector in the desired direction theta
   s = s.reshape(-1,1) # make into a column vector (size 3x1)
   R = (X @ X.conj().T)/X.shape[1] # Calc covariance matrix. gives a Nr x Nr covariance matrix of the samples
   Rinv = np.linalg.pinv(R) # 3x3. pseudo-inverse tends to work better/faster than a true inverse
   w = (Rinv @ s)/(s.conj().T @ Rinv @ s) # MVDR/Capon equation! numerator is 3x3 * 3x1, denominator is 1x3 * 3x3 * 3x1, resulting in a 3x1 weights vector
   return w


theta_scan = np.linspace(-1*np.pi, np.pi, 1000) # 1000 different thetas between -180 and +180 degrees
results = []
for theta_i in theta_scan:
   w = w_mvdr(theta_i, X) # 3x1
   X_weighted = w.conj().T @ X # apply weights
   power_dB = 10*np.log10(np.var(X_weighted)) # power in signal, in dB so its easier to see small and large lobes at the same time
   results.append(power_dB)
results -= np.max(results) # normalize
mvdr_A = results
conv_A = conventional_scan(theta_scan, X)

# Waveforms: the three noisy inputs and the beamformer outputs steered at 20 deg
w_conv = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta)).reshape(-1,1) / Nr # /Nr for unit gain at look direction
y_conv = (w_conv.conj().T @ X).squeeze()
y_mvdr = (w_mvdr(theta, X).conj().T @ X).squeeze()  # MVDR has unit gain at look direction by construction
tx_A = tx.squeeze()
M = 200
fig_w, axes_w = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
axes_w[0].plot(tx_A[:M].real, "k")
axes_w[0].set_title("Part A: transmitted tone (clean)")
for k in range(Nr):
    axes_w[1].plot(X[k, :M].real, lw=0.8, label=f"element {k}")
axes_w[1].set_title("Received at each element (tone from 20° + noise 0.1)")
axes_w[1].legend(loc="upper right")
axes_w[2].plot(tx_A[:M].real, "k", lw=1, label="clean tone")
axes_w[2].plot(y_conv[:M].real, "C0", lw=1, label="conventional output (wᴴX / Nr)")
axes_w[2].plot(y_mvdr[:M].real, "C1", lw=1, linestyle="--", label="MVDR output (w_mvdrᴴ X)")
axes_w[2].set_title("Beamformer outputs steered to 20°")
axes_w[2].legend(loc="upper right")
axes_w[2].set_xlabel("sample")
for a in axes_w:
    a.grid(alpha=0.3)
fig_w.tight_layout()
os.makedirs(PLOTS_DIR, exist_ok=True)
fig_w.savefig(os.path.join(PLOTS_DIR, "pysdr_mvdr_waveforms.png"), dpi=130)
plt.close(fig_w)
for name, y in (("element 0", X[0]), ("conventional", y_conv), ("MVDR", y_mvdr)):
    err = y - tx_A
    print("Part A  SNR %-12s %6.2f dB" % (name, 10*np.log10(np.mean(np.abs(tx_A)**2)/np.mean(np.abs(err)**2))))
print("Part A  MVDR peak: %.2f deg   conventional peak: %.2f deg"
      % (theta_scan[np.argmax(mvdr_A)] * 180 / np.pi,
         theta_scan[np.argmax(conv_A)] * 180 / np.pi))

# --------------------------------------------------------------------------
# Part B: Nr = 8, three sources
# --------------------------------------------------------------------------
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


def power_mvdr(theta, X):
    s = np.exp(2j * np.pi * d * np.arange(X.shape[0]) * np.sin(theta)) # steering vector in the desired direction theta_i
    s = s.reshape(-1,1) # make into a column vector (size 3x1)
    R = (X @ X.conj().T)/X.shape[1] # Calc covariance matrix. gives a Nr x Nr covariance matrix of the samples
    Rinv = np.linalg.pinv(R) # 3x3. pseudo-inverse tends to work better than a true inverse
    return 1/(s.conj().T @ Rinv @ s).squeeze()


results = []
for theta_i in theta_scan:
   power_dB = 10*np.log10(np.abs(power_mvdr(theta_i, X)))
   results.append(power_dB)
results -= np.max(results) # normalize
mvdr_B = results
conv_B = conventional_scan(theta_scan, X)

# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
deg = theta_scan * 180 / np.pi
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

ax = axes[0, 0]
ax.plot(deg, conv_A, label="conventional")
ax.plot(deg, mvdr_A, label="MVDR (w_mvdr, applied to X)")
ax.axvline(theta_degrees, color="k", linewidth=0.8, linestyle=":")
ax.set_xlim(-90, 90); ax.set_xlabel("Theta [Degrees]"); ax.set_ylabel("DOA Metric [dB]")
ax.set_title("Part A: Nr=3, one source at 20°"); ax.grid(); ax.legend()

ax = fig.add_subplot(2, 2, 2, projection="polar"); axes[0, 1].remove()
ax.plot(theta_scan, np.maximum(conv_A, -40), label="conventional")
ax.plot(theta_scan, np.maximum(mvdr_A, -40), label="MVDR")
ax.set_theta_zero_location('N') # make 0 degrees point up
ax.set_theta_direction(-1) # increase clockwise
ax.set_rlabel_position(55)  # Move grid labels away from other labels
ax.set_rlim(-40, 0); ax.set_title("Part A, polar"); ax.legend(loc="lower left")

ax = axes[1, 0]
ax.plot(deg, conv_B, label="conventional")
ax.plot(deg, mvdr_B, label="MVDR (power_mvdr = 1 / sᴴR⁻¹s)")
for a in (20, 25, -40):
    ax.axvline(a, color="k", linewidth=0.8, linestyle=":")
ax.set_xlim(-90, 90); ax.set_xlabel("Theta [Degrees]"); ax.set_ylabel("DOA Metric [dB]")
ax.set_title("Part B: Nr=8, sources at 20°, 25° and −40° (−20 dB)"); ax.grid(); ax.legend()

ax = axes[1, 1]
ax.plot(deg, conv_B, label="conventional")
ax.plot(deg, mvdr_B, label="MVDR")
for a in (20, 25):
    ax.axvline(a, color="k", linewidth=0.8, linestyle=":")
ax.set_xlim(10, 35); ax.set_ylim(-25, 1)
ax.set_xlabel("Theta [Degrees]"); ax.set_title("Part B, zoom on 20° / 25°"); ax.grid(); ax.legend()

fig.tight_layout()
os.makedirs(PLOTS_DIR, exist_ok=True)
out = os.path.join(PLOTS_DIR, "pysdr_mvdr.png")
fig.savefig(out, dpi=130)
print(f"wrote {out}")
