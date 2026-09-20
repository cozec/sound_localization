"""PySDR "LMS" section as a standalone script.

Code from https://pysdr.org/content/doa.html (Marc Lichtman, PySDR,
CC BY-NC-SA 4.0). Comments are the original author's; plt.show() replaced by
savefig, plus the beam pattern the page says to "plot as shown previously"
and a check against MVDR steered to the true SOI direction.

LMS (least mean squares) is the odd one out: it does NOT know the DOA. It
knows the *waveform* of the signal of interest (here a repeated Gold code,
i.e. a training / pilot sequence) and adapts the weights sample by sample to
make the beamformer output match it:
    y = w^H r,  e = soi - y,  w <- w + mu * conj(e) * r
The steering toward 20 deg and the nulls on the two tone jammers at 60 deg
and -50 deg emerge from the data alone.

Output: plots/pysdr_lms.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")

np.random.seed(0)

# Scenario
sample_rate = 1e6
d = 0.5 # half wavelength spacing
N = 100000 # number of samples to simulate
Nr = 8 # elements
theta_soi = 20 / 180 * np.pi # convert to radians
theta2    = 60 / 180 * np.pi
theta3   = -50 / 180 * np.pi
t = np.arange(N)/sample_rate # time vector
s1 = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta_soi)).reshape(-1,1) # 8x1
s2 = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta2)).reshape(-1,1)
s3 = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta3)).reshape(-1,1)

# SOI is a gold code, repeated, length 127
gold_code = np.array([-1, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1])
soi_samples_per_symbol = 8
soi = np.repeat(gold_code, soi_samples_per_symbol)
num_sequence_repeats = int(N / soi.shape[0]) + 1 # number of times to repeat the sequence for N samples
soi = np.tile(soi, num_sequence_repeats)[:N] # repeat the sequence to fill simulated time, then trim
soi = soi.reshape(1, -1) # 1xN

# Interference, eg tone jammers, from different directions
tone2 = np.exp(2j*np.pi*0.02e6*t).reshape(1,-1)
tone3 = np.exp(2j*np.pi*0.03e6*t).reshape(1,-1)

# Simulate received signal
r = s1 @ soi + s2 @ tone2 + s3 @ tone3
n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
r = r + 0.5*n # 8xN

# LMS, not knowing the direction of SOI but knowing the SOI signal itself
mu = 0.5e-5 # LMS step size
w_lms = np.zeros((Nr, 1), dtype=np.complex128) # start with all zeros

# Loop through received samples
error_log = []
for i in range(N):
   r_sample = r[:, i].reshape(-1, 1) # 8x1
   soi_sample = soi[0, i] # scalar
   y = w_lms.conj().T @ r_sample # apply the weights
   y = y.squeeze() # make it a scalar
   error = soi_sample - y
   error_log.append(np.abs(error)**2)
   w_lms += mu * np.conj(error) * r_sample # weights are still 8x1

w_lms_unnorm = w_lms.copy()          # keep the converged (unit-gain) weights for the waveform plot
w_lms /= np.linalg.norm(w_lms) # normalize weights

# --------------------------------------------------------------------------
# Compare with MVDR steered at the (here known) SOI direction
# --------------------------------------------------------------------------
R = (r @ r.conj().T) / N
Rinv = np.linalg.pinv(R)
w_mvdr = (Rinv @ s1) / (s1.conj().T @ Rinv @ s1)


def beam_pattern_db(w, N_fft=1024):
    w = w.squeeze()
    w_padded = np.concatenate((w, np.zeros(N_fft - Nr))) # zero pad to N_fft elements to get more resolution in the FFT
    w_fft_dB = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(w_padded)))**2) # magnitude of fft in dB
    w_fft_dB -= np.max(w_fft_dB) # normalize to 0 dB at peak
    theta_bins = np.arcsin(np.linspace(-1, 1, N_fft)) # Map the FFT bins to angles in radians
    return theta_bins, w_fft_dB


np.set_printoptions(precision=4, suppress=True)
print("w_lms (normalized)  =", w_lms.squeeze())
print("w_mvdr (normalized) =", (w_mvdr / np.linalg.norm(w_mvdr)).squeeze())
for name, w in (("LMS", w_lms_unnorm), ("MVDR", w_mvdr)):
    gs = [abs((w.conj().T @ s).squeeze()) for s in (s1, s2, s3)]
    print(f"{name:5s} gain @SOI 20°: {gs[0]:.4f} ({20*np.log10(gs[0]):+.1f} dB)   "
          f"@60°: {gs[1]:.4f} ({20*np.log10(gs[1]):+.1f} dB)   @-50°: {gs[2]:.4f} ({20*np.log10(gs[2]):+.1f} dB)")

y_lms = (w_lms_unnorm.conj().T @ r).squeeze()
y_mvdr = (w_mvdr.conj().T @ r).squeeze()
soi_1d = soi.squeeze()
tail = slice(N - 20000, N)   # after convergence
for name, y in (("element 0", r[0]), ("LMS", y_lms), ("MVDR", y_mvdr)):
    err = y[tail] - soi_1d[tail]
    print("SNR %-10s %6.2f dB (last 20k samples)" % (name, 10*np.log10(np.mean(np.abs(soi_1d[tail])**2)/np.mean(np.abs(err)**2))))

# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 9))

ax = fig.add_subplot(2, 2, 1)
ax.plot(error_log, lw=0.5)
ax.set_xlabel('Iteration')
ax.set_ylabel('Mean Square Error')
ax.set_yscale("log")
ax.set_title(f"LMS learning curve (mu = {mu:g})")
ax.grid(alpha=0.3)

ax = fig.add_subplot(2, 2, 2, projection='polar')
th, p = beam_pattern_db(w_lms)
ax.plot(th, p, label="LMS (no DOA given)")
th, p = beam_pattern_db(w_mvdr)
ax.plot(th, p, "--", label="MVDR steered to 20°")
ax.plot([theta_soi], [0], 'og')
ax.plot([theta2], [0], 'or')
ax.plot([theta3], [0], 'or')
ax.set_theta_zero_location('N') # make 0 degrees point up
ax.set_theta_direction(-1) # increase clockwise
ax.set_thetagrids(np.arange(-90, 105, 15)) # it's in degrees
ax.set_rlabel_position(55)  # Move grid labels away from other labels
ax.set_thetamin(-90) # only show top half
ax.set_thetamax(90)
ax.set_ylim([-40, 1])
ax.set_title("Beam pattern: SOI 20° (green), jammers 60° / −50° (red)", pad=18)
ax.legend(loc="lower left", fontsize=8)

M0, M1 = N - 600, N - 600 + 300
ax = fig.add_subplot(2, 1, 2)
ax.plot(soi_1d[M0:M1].real, "k", lw=1.2, label="SOI (Gold code, ±1)")
ax.plot(r[0, M0:M1].real, "C7", lw=0.6, alpha=0.7, label="element 0 (raw)")
ax.plot(y_lms[M0:M1].real, "C0", lw=1, label="LMS output")
ax.plot(y_mvdr[M0:M1].real, "C1", lw=1, ls="--", label="MVDR output")
ax.set_xlabel(f"sample (offset {M0})")
ax.set_title("Recovered SOI after convergence")
ax.legend(loc="upper right", ncol=4, fontsize=9)
ax.grid(alpha=0.3)

fig.tight_layout()
os.makedirs(PLOTS_DIR, exist_ok=True)
out = os.path.join(PLOTS_DIR, "pysdr_lms.png")
fig.savefig(out, dpi=130)
print(f"wrote {out}")
