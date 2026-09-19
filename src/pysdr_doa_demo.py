"""PySDR "Beamforming & DOA" chapter demo, stitched into one runnable script.

Code is taken from https://pysdr.org/content/doa.html (Marc Lichtman,
PySDR, CC BY-NC-SA 4.0) with minimal edits: plt.show() replaced by savefig,
sections wired together so they run top to bottom. Comments are the
original author's. Full snippet dump: reference/pysdr_doa_snippets.md.

RF conventions: d is element spacing in wavelengths (0.5 = lambda/2), Nr is
the number of receive elements, steering vector exp(2j*pi*d*k*sin(theta)).
The math is identical for a narrowband mic array with d = spacing*f/c.

Sections:
  1. Simulate a 3-element ULA receiving a tone from 20 deg
  2. Conventional (delay-and-sum) DOA scan, cartesian + polar
  3. Beam pattern of the conventional beamformer via zero-padded FFT
  4. MVDR/Capon DOA scan, 8 elements, three sources (20, 25, -40 deg)
  5. MUSIC pseudo-spectrum on the same scenario
  6. Root MUSIC closed-form DOAs

Output: plots/pysdr_doa_demo.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")

np.random.seed(0)

# --------------------------------------------------------------------------
# 1. Receiving a Signal
# --------------------------------------------------------------------------
sample_rate = 1e6
N = 10000 # number of samples to simulate

# Create a tone to act as the transmitter signal
t = np.arange(N)/sample_rate # time vector
f_tone = 0.02e6
tx = np.exp(2j * np.pi * f_tone * t)

d = 0.5 # half wavelength spacing
Nr = 3
theta_degrees = 20 # direction of arrival (feel free to change this, it's arbitrary)
theta = theta_degrees / 180 * np.pi # convert to radians
s = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta)) # Steering Vector
print(s) # note that it's 3 elements long, it's complex, and the first element is 1+0j

s = s.reshape(-1,1) # make s a column vector
tx = tx.reshape(1,-1) # make tx a row vector
X = s @ tx # Simulate the received signal X through a matrix multiply
print(X.shape) # 3x10000.  X is now going to be a 2D array, 1D is time and 1D is the spatial dimension

fig_rx, ax_rx = plt.subplots(figsize=(8, 3))
for k in range(Nr):
    ax_rx.plot(X[k, :200].real, label=f"element {k}") # X is a plain ndarray here, no asarray/squeeze needed
ax_rx.set_xlabel("sample")
ax_rx.set_ylabel("real part")
ax_rx.set_title(f"Received tone at each element, source at {theta_degrees}° (phase shift = steering vector)")
ax_rx.legend(loc="upper right")
ax_rx.grid()
fig_rx.tight_layout()
os.makedirs(PLOTS_DIR, exist_ok=True)
fig_rx.savefig(os.path.join(PLOTS_DIR, "pysdr_received_signal.png"), dpi=130)
plt.close(fig_rx)

n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
X = X + 0.1*n # X and n are both 3x10000

# --------------------------------------------------------------------------
# 2. Conventional Beamforming & DOA
# --------------------------------------------------------------------------
theta_scan = np.linspace(-1*np.pi, np.pi, 1000) # 1000 different thetas between -180 and +180 degrees
results = []
for theta_i in theta_scan:
   w = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta_i)) # Conventional, aka delay-and-sum, beamformer
   X_weighted = w.conj().T @ X # apply our weights. remember X is 3x10000
   results.append(10*np.log10(np.var(X_weighted))) # power in signal, in dB so its easier to see small and large lobes at the same time
results -= np.max(results) # normalize (optional)
conv_results = results

# print angle that gave us the max value
print("conventional DOA:", theta_scan[np.argmax(results)] * 180 / np.pi) # 19.99999999999998

# --------------------------------------------------------------------------
# 3. Beam Pattern
# --------------------------------------------------------------------------
N_fft = 512
w = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta)) # conventional beamformer
w_padded = np.concatenate((w, np.zeros(N_fft - Nr))) # zero pad to N_fft elements to get more resolution in the FFT
w_fft_dB = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(w_padded)))**2) # magnitude of fft in dB
w_fft_dB -= np.max(w_fft_dB) # normalize to 0 dB at peak

# Map the FFT bins to angles in radians
theta_bins = np.arcsin(np.linspace(-1, 1, N_fft)) # in radians

# find max so we can add it to plot
theta_max = theta_bins[np.argmax(w_fft_dB)]

# --------------------------------------------------------------------------
# 4. MVDR/Capon Beamformer — 8 elements, three sources
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


# Conventional scan on the 8-element scenario for comparison
results = []
for theta_i in theta_scan:
   w = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta_i))
   X_weighted = w.conj().T @ X
   results.append(10*np.log10(np.var(X_weighted)))
results -= np.max(results)
conv8_results = results

results = []
for theta_i in theta_scan:
   power_dB = 10*np.log10(np.abs(power_mvdr(theta_i, X)))
   results.append(power_dB)
results -= np.max(results) # normalize
mvdr_results = results

# --------------------------------------------------------------------------
# 5. MUSIC
# --------------------------------------------------------------------------
num_expected_signals = 3 # Try changing this!

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

results -= np.max(results) # normalize (dB offset; original divides, which is wrong for dB)
music_results = results

# --------------------------------------------------------------------------
# 6. Root MUSIC
# --------------------------------------------------------------------------
V = v[:, :Nr - num_expected_signals]  # noise subspace eigenvectors

# Build the Root MUSIC polynomial from diagonals of noise-subspace projection
D = V @ V.conj().T
p = np.zeros(2*Nr - 1, dtype=np.complex128)
for k in range(2*Nr - 1):
    p[k] = np.sum(np.diag(D, k - (Nr - 1)))

# Find roots, keep those inside the unit circle, pick the num_expected_signals roots closest to the unit circle
roots = np.roots(p[::-1])  # np.roots expects highest-degree coefficient first
roots = roots[np.abs(roots) <= 1.0] # remove the conjugate-reciprocal partners which correspond to the same DOA estimate anyway
roots = roots[np.argsort(-np.abs(roots))]  # sort closest-to-unit-circle first
doa_roots = roots[:num_expected_signals]

# Convert roots to angles in degrees
doas_deg = np.sort(np.arcsin(np.angle(doa_roots) / (2 * np.pi * d)) * 180 / np.pi)
print("Root MUSIC estimated DOAs (degrees):", doas_deg)

# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 9))

ax = fig.add_subplot(2, 3, 1)
ax.plot(theta_scan*180/np.pi, conv_results) # lets plot angle in degrees
ax.set_xlabel("Theta [Degrees]")
ax.set_ylabel("DOA Metric")
ax.set_title("Conventional DOA scan, Nr=3, source at 20°")
ax.grid()

ax = fig.add_subplot(2, 3, 2, projection='polar')
ax.plot(theta_scan, conv_results) # MAKE SURE TO USE RADIAN FOR POLAR
ax.set_theta_zero_location('N') # make 0 degrees point up
ax.set_theta_direction(-1) # increase clockwise
ax.set_rlabel_position(55)  # Move grid labels away from other labels
ax.set_title("Same scan, polar")

ax = fig.add_subplot(2, 3, 3, projection='polar')
ax.plot(theta_bins, w_fft_dB) # MAKE SURE TO USE RADIAN FOR POLAR
ax.plot([theta_max], [np.max(w_fft_dB)],'ro')
ax.text(theta_max - 0.1, np.max(w_fft_dB) - 4, np.round(theta_max * 180 / np.pi))
ax.set_theta_zero_location('N') # make 0 degrees point up
ax.set_theta_direction(-1) # increase clockwise
ax.set_rlabel_position(55)  # Move grid labels away from other labels
ax.set_thetamin(-90) # only show top half
ax.set_thetamax(90)
ax.set_ylim([-30, 1]) # because there's no noise, only go down 30 dB
ax.set_title("Beam pattern (FFT of weights), steered 20°")

ax = fig.add_subplot(2, 1, 2)
ax.plot(theta_scan*180/np.pi, conv8_results, label="conventional")
ax.plot(theta_scan*180/np.pi, mvdr_results, label="MVDR")
ax.plot(theta_scan*180/np.pi, music_results, label="MUSIC")
for a in (20, 25, -40):
    ax.axvline(a, color="k", linewidth=0.8, linestyle=":")
ax.set_xlim(-90, 90)
ax.set_xlabel("Theta [Degrees]")
ax.set_ylabel("DOA Metric [dB]")
ax.set_title("Nr=8: sources at 20°, 25° and −40° (−20 dB) — "
             f"Root MUSIC: {np.round(doas_deg, 2)}")
ax.legend()
ax.grid()

fig.tight_layout()
os.makedirs(PLOTS_DIR, exist_ok=True)
out = os.path.join(PLOTS_DIR, "pysdr_doa_demo.png")
fig.savefig(out, dpi=130)
print(f"wrote {out}")
