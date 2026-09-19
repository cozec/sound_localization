"""PySDR "LCMV Beamformer" section as a standalone script.

Code from https://pysdr.org/content/doa.html (Marc Lichtman, PySDR,
CC BY-NC-SA 4.0). Comments are the original author's; plt.show() replaced
by savefig and the two examples put side by side.

LCMV generalizes MVDR from one constraint (w^H s = 1) to a set:
    C^H w = f,  C = [s_1 ... s_K],  f = desired gains
    w = R^-1 C (C^H R^-1 C)^-1 f

Example 1: 8 elements, four equal-power interferers at -60, -30, 0, +30 deg,
           noise 0.5. Constraints: unit gain at 15 deg and 60 deg (two SOIs,
           f = [1, 1]). Nothing was transmitted from those angles; the point
           is that the interferers get nulled while both look directions
           are held at 0 dB.
Example 2: 18 elements, noise only (R = I). Unit gain across 15..30 deg
           (4 constraints, f = 1) and a null across 45..60 deg (4 constraints,
           f = 0) -- a wide passband and a wide stopband from one weight vector.

Output: plots/pysdr_lcmv.png
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


def beam_pattern_db(w, N_fft=1024):
    """Zero-padded FFT of the weights mapped to angle (from the "Beam Pattern" section)."""
    Nr = len(w)
    w_padded = np.concatenate((w, np.zeros(N_fft - Nr))) # zero pad to N_fft elements to get more resolution in the FFT
    w_fft_dB = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(w_padded)))**2) # magnitude of fft in dB
    w_fft_dB -= np.max(w_fft_dB) # normalize to 0 dB at peak
    theta_bins = np.arcsin(np.linspace(-1, 1, N_fft)) # Map the FFT bins to angles in radians
    return theta_bins, w_fft_dB


def style_polar(ax, title):
    ax.set_theta_zero_location('N') # make 0 degrees point up
    ax.set_theta_direction(-1) # increase clockwise
    ax.set_thetagrids(np.arange(-90, 105, 15)) # it's in degrees
    ax.set_rlabel_position(55)  # Move grid labels away from other labels
    ax.set_thetamin(-90) # only show top half
    ax.set_thetamax(90)
    ax.set_ylim([-30, 1]) # because there's no noise, only go down 30 dB
    ax.set_title(title, pad=18)


# --------------------------------------------------------------------------
# Example 1: four interferers, two SOI constraints
# --------------------------------------------------------------------------
# Simulate received signal
Nr = 8 # 8 elements
theta1 = -60 / 180 * np.pi # convert to radians
theta2 = -30 / 180 * np.pi
theta3 = 0 / 180 * np.pi
theta4 = 30 / 180 * np.pi
s1 = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta1)).reshape(-1,1) # 8x1
s2 = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta2)).reshape(-1,1)
s3 = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta3)).reshape(-1,1)
s4 = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta4)).reshape(-1,1)
# we'll use 3 different frequencies.  1xN
tone1 = np.exp(2j*np.pi*0.01e6*t).reshape(1,-1)
tone2 = np.exp(2j*np.pi*0.02e6*t).reshape(1,-1)
tone3 = np.exp(2j*np.pi*0.03e6*t).reshape(1,-1)
tone4 = np.exp(2j*np.pi*0.04e6*t).reshape(1,-1)
X = s1 @ tone1 + s2 @ tone2 + s3 @ tone3 + s4 @ tone4
n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
X = X + 0.5*n # 8xN

# Let's point at the SOI at 15 deg, and another potential SOI that we didn't actually simulate at 60 deg
soi1_theta = 15 / 180 * np.pi # convert to radians
soi2_theta = 60 / 180 * np.pi

# LCMV weights
R_inv = np.linalg.pinv(np.cov(X)) # 8x8
s1 = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(soi1_theta)).reshape(-1,1) # 8x1
s2 = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(soi2_theta)).reshape(-1,1) # 8x1
C = np.concatenate((s1, s2), axis=1) # 8x2
f = np.ones(2).reshape(-1,1) # 2x1

# LCMV equation
#    8x8   8x2                    2x8        8x8   8x2  2x1
w = R_inv @ C @ np.linalg.pinv(C.conj().T @ R_inv @ C) @ f # output is 8x1

w = w.squeeze() # reduce to a 1D array
theta_bins, w_fft_dB = beam_pattern_db(w)

# Check the constraints and the interferer gains directly
np.set_printoptions(precision=4, suppress=True)
print("Example 1  w =", w)
for name, th in (("SOI 15", soi1_theta), ("SOI 60", soi2_theta), ("int -60", theta1),
                 ("int -30", theta2), ("int 0", theta3), ("int 30", theta4)):
    s = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(th))
    g = abs(w.conj() @ s)
    print(f"  gain at {name:8s}: {g:.4f}  ({20*np.log10(g):+.1f} dB)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5), subplot_kw={'projection': 'polar'})

ax1.plot(theta_bins, w_fft_dB) # MAKE SURE TO USE RADIAN FOR POLAR
# Add dots where interferers and SOIs are
ax1.plot([theta1], [0], 'or')
ax1.plot([theta2], [0], 'or')
ax1.plot([theta3], [0], 'or')
ax1.plot([theta4], [0], 'or')
ax1.plot([soi1_theta], [0], 'og')
ax1.plot([soi2_theta], [0], 'og')
style_polar(ax1, "Ex 1: Nr=8, SOIs at 15° and 60° (green, f=1)\ninterferers at −60°, −30°, 0°, 30° (red)")

# --------------------------------------------------------------------------
# Example 2: wide passband + wide null, noise-only R
# --------------------------------------------------------------------------
Nr = 18
X = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N) # Simulate received signal of just noise

# Let's point at the SOI from 15 to 30 degrees using 4 different thetas
soi_thetas = np.linspace(15, 30, 4) / 180 * np.pi # convert to radians

# Let's make a null from 45 to 60 degrees using 4 different thetas
null_thetas = np.linspace(45, 60, 4) / 180 * np.pi # convert to radians

# LCMV weights
R_inv = np.linalg.pinv(np.cov(X))
s = []
for soi_theta in soi_thetas:
    s.append(np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(soi_theta)).reshape(-1,1))
for null_theta in null_thetas:
    s.append(np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(null_theta)).reshape(-1,1))
C = np.concatenate(s, axis=1)
f = np.asarray([1]*len(soi_thetas) + [0]*len(null_thetas)).reshape(-1,1)
w = R_inv @ C @ np.linalg.pinv(C.conj().T @ R_inv @ C) @ f # LCMV equation

# Plot beam pattern as before...
w = w.squeeze()
theta_bins, w_fft_dB = beam_pattern_db(w)
ax2.plot(theta_bins, w_fft_dB)
ax2.plot(soi_thetas, np.zeros(len(soi_thetas)), 'og')
ax2.plot(null_thetas, np.zeros(len(null_thetas)), 'or')
style_polar(ax2, "Ex 2: Nr=18, noise-only R\npassband 15–30° (green, f=1), null 45–60° (red, f=0)")

print("Example 2  gain across passband / stopband:")
for th in np.concatenate((soi_thetas, null_thetas)):
    s = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(th))
    g = abs(w.conj() @ s)
    print(f"  {np.degrees(th):5.1f}°: {g:.4f}  ({20*np.log10(max(g, 1e-9)):+.1f} dB)")

fig.tight_layout()
os.makedirs(PLOTS_DIR, exist_ok=True)
out = os.path.join(PLOTS_DIR, "pysdr_lcmv.png")
fig.savefig(out, dpi=130)
print(f"wrote {out}")
