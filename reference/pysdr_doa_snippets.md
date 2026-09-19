# PySDR — Beamforming & DOA: code snippets

Pulled verbatim from https://pysdr.org/content/doa.html (Marc Lichtman, PySDR, CC BY-NC-SA 4.0).
Snippets are in page order, grouped by the section heading they appear under.
RF conventions: `d` is element spacing in wavelengths (0.5 = λ/2), `Nr` = number of elements,
steering vector `exp(2j*pi*d*k*sin(theta))`. For our mic array, `d` = spacing / λ = spacing·f / c.


## Intro to Matrix Math in Python/NumPy

```python
A = np.random.randn(3,10) # 3x10
B = np.arange(10) # 1D array of length 10
B = B.reshape(-1,1) # 10x1
C = A @ B # matrix multiply
print(C.shape) # 3x1
C = C.squeeze() # see next subsection
print(C.shape) # 1D array of length 3, easier for plotting and other non-matrix Python code
```


## Steering Vector

```python
s = [np.exp(2j*np.pi*d*0*np.sin(theta)), np.exp(2j*np.pi*d*1*np.sin(theta)), np.exp(2j*np.pi*d*2*np.sin(theta)), ...] # note the increasing k
# or
s = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta)) # where Nr is the number of receive antenna elements
```


## Receiving a Signal

```python
import numpy as np
import matplotlib.pyplot as plt

sample_rate = 1e6
N = 10000 # number of samples to simulate

# Create a tone to act as the transmitter signal
t = np.arange(N)/sample_rate # time vector
f_tone = 0.02e6
tx = np.exp(2j * np.pi * f_tone * t)
```

```python
d = 0.5 # half wavelength spacing
Nr = 3
theta_degrees = 20 # direction of arrival (feel free to change this, it's arbitrary)
theta = theta_degrees / 180 * np.pi # convert to radians
s = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta)) # Steering Vector
print(s) # note that it's 3 elements long, it's complex, and the first element is 1+0j
```

```python
s = s.reshape(-1,1) # make s a column vector
print(s.shape) # 3x1
tx = tx.reshape(1,-1) # make tx a row vector
print(tx.shape) # 1x10000

X = s @ tx # Simulate the received signal X through a matrix multiply
print(X.shape) # 3x10000.  X is now going to be a 2D array, 1D is time and 1D is the spatial dimension
```

```python
plt.plot(np.asarray(X[0,:]).squeeze().real[0:200]) # the asarray and squeeze are just annoyances we have to do because we came from a matrix
plt.plot(np.asarray(X[1,:]).squeeze().real[0:200])
plt.plot(np.asarray(X[2,:]).squeeze().real[0:200])
plt.show()
```

```python
n = np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N)
X = X + 0.1*n # X and n are both 3x10000
```


## Conventional Beamforming & DOA

```python
w = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta)) # Conventional, aka delay-and-sum, beamformer
X_weighted = w.conj().T @ X # example of applying the weights to the received signal (i.e., perform the beamforming)
print(X_weighted.shape) # 1x10000
```

```python
theta_scan = np.linspace(-1*np.pi, np.pi, 1000) # 1000 different thetas between -180 and +180 degrees
results = []
for theta_i in theta_scan:
   w = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta_i)) # Conventional, aka delay-and-sum, beamformer
   X_weighted = w.conj().T @ X # apply our weights. remember X is 3x10000
   results.append(10*np.log10(np.var(X_weighted))) # power in signal, in dB so its easier to see small and large lobes at the same time
results -= np.max(results) # normalize (optional)

# print angle that gave us the max value
print(theta_scan[np.argmax(results)] * 180 / np.pi) # 19.99999999999998

plt.plot(theta_scan*180/np.pi, results) # lets plot angle in degrees
plt.xlabel("Theta [Degrees]")
plt.ylabel("DOA Metric")
plt.grid()
plt.show()
```

```python
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta_scan, results) # MAKE SURE TO USE RADIAN FOR POLAR
ax.set_theta_zero_location('N') # make 0 degrees point up
ax.set_theta_direction(-1) # increase clockwise
ax.set_rlabel_position(55)  # Move grid labels away from other labels
plt.show()
```


## Beam Pattern

```python
np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta))
```

```python
Nr = 3
d = 0.5
N_fft = 512
theta_degrees = 20 # there is no SOI, we arent processing samples, this is just the direction we want to point at
theta = theta_degrees / 180 * np.pi
w = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta)) # conventional beamformer
w_padded = np.concatenate((w, np.zeros(N_fft - Nr))) # zero pad to N_fft elements to get more resolution in the FFT
w_fft_dB = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(w_padded)))**2) # magnitude of fft in dB
w_fft_dB -= np.max(w_fft_dB) # normalize to 0 dB at peak

# Map the FFT bins to angles in radians
theta_bins = np.arcsin(np.linspace(-1, 1, N_fft)) # in radians

# find max so we can add it to plot
theta_max = theta_bins[np.argmax(w_fft_dB)]

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta_bins, w_fft_dB) # MAKE SURE TO USE RADIAN FOR POLAR
ax.plot([theta_max], [np.max(w_fft_dB)],'ro')
ax.text(theta_max - 0.1, np.max(w_fft_dB) - 4, np.round(theta_max * 180 / np.pi))
ax.set_theta_zero_location('N') # make 0 degrees point up
ax.set_theta_direction(-1) # increase clockwise
ax.set_rlabel_position(55)  # Move grid labels away from other labels
ax.set_thetamin(-90) # only show top half
ax.set_thetamax(90)
ax.set_ylim([-30, 1]) # because there's no noise, only go down 30 dB
plt.show()
```


## Spatial Tapering

```python
tapering = np.random.uniform(0, 1, Nr) # random tapering
w *= tapering
```

```python
tapering = np.hamming(Nr) # Hamming window function
w *= tapering
```


## MVDR/Capon Beamformer

```python
# theta is the direction of interest, in radians, and X is our received signal
def w_mvdr(theta, X):
   s = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta)) # steering vector in the desired direction theta
   s = s.reshape(-1,1) # make into a column vector (size 3x1)
   R = (X @ X.conj().T)/X.shape[1] # Calc covariance matrix. gives a Nr x Nr covariance matrix of the samples
   Rinv = np.linalg.pinv(R) # 3x3. pseudo-inverse tends to work better/faster than a true inverse
   w = (Rinv @ s)/(s.conj().T @ Rinv @ s) # MVDR/Capon equation! numerator is 3x3 * 3x1, denominator is 1x3 * 3x3 * 3x1, resulting in a 3x1 weights vector
   return w
```

```python
theta_scan = np.linspace(-1*np.pi, np.pi, 1000) # 1000 different thetas between -180 and +180 degrees
results = []
for theta_i in theta_scan:
   w = w_mvdr(theta_i, X) # 3x1
   X_weighted = w.conj().T @ X # apply weights
   power_dB = 10*np.log10(np.var(X_weighted)) # power in signal, in dB so its easier to see small and large lobes at the same time
   results.append(power_dB)
results -= np.max(results) # normalize
```

```python
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
```

```python
def power_mvdr(theta, X):
    s = np.exp(2j * np.pi * d * np.arange(X.shape[0]) * np.sin(theta)) # steering vector in the desired direction theta_i
    s = s.reshape(-1,1) # make into a column vector (size 3x1)
    R = (X @ X.conj().T)/X.shape[1] # Calc covariance matrix. gives a Nr x Nr covariance matrix of the samples
    Rinv = np.linalg.pinv(R) # 3x3. pseudo-inverse tends to work better than a true inverse
    return 1/(s.conj().T @ Rinv @ s).squeeze()
```


## Covariance Matrix

```python
[[ 1.494+0.j    0.486+0.881j -0.543+0.839j]
 [ 0.486-0.881j 1.517 +0.j    0.483+0.886j]
 [-0.543-0.839j 0.483-0.886j  1.499+0.j   ]]
```


## LCMV Beamformer

```python
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
```

```python
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

# Plot beam pattern
w = w.squeeze() # reduce to a 1D array
N_fft = 1024
w_padded = np.concatenate((w, np.zeros(N_fft - Nr))) # zero pad to N_fft elements to get more resolution in the FFT
w_fft_dB = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(w_padded)))**2) # magnitude of fft in dB
w_fft_dB -= np.max(w_fft_dB) # normalize to 0 dB at peak
theta_bins = np.arcsin(np.linspace(-1, 1, N_fft)) # Map the FFT bins to angles in radians

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta_bins, w_fft_dB) # MAKE SURE TO USE RADIAN FOR POLAR
# Add dots where interferers and SOIs are
ax.plot([theta1], [0], 'or')
ax.plot([theta2], [0], 'or')
ax.plot([theta3], [0], 'or')
ax.plot([theta4], [0], 'or')
ax.plot([soi1_theta], [0], 'og')
ax.plot([soi2_theta], [0], 'og')
ax.set_theta_zero_location('N') # make 0 degrees point up
ax.set_theta_direction(-1) # increase clockwise
ax.set_thetagrids(np.arange(-90, 105, 15)) # it's in degrees
ax.set_rlabel_position(55)  # Move grid labels away from other labels
ax.set_thetamin(-90) # only show top half
ax.set_thetamax(90)
ax.set_ylim([-30, 1]) # because there's no noise, only go down 30 dB
plt.show()
```

```python
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
```


## Null Steering

```python
d = 0.5
Nr = 8

theta_soi = 30 / 180 * np.pi # convert to radians
nulls_deg = [-60, -30, 0, 60] # degrees
nulls_rad = np.asarray(nulls_deg) / 180 * np.pi

# Start out with conventional beamformer pointed at theta_soi
w = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta_soi)).reshape(-1,1)

# Loop through nulls
for null_rad in nulls_rad:
    # weights equal to steering vector in target null direction
    w_null = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(null_rad)).reshape(-1,1)

    # scaling_factor (complex scalar) for w at nulled direction
    scaling_factor = w_null.conj().T @ w / (w_null.conj().T @ w_null)
    print("scaling_factor:", scaling_factor, scaling_factor.shape)

    # Update weights to include the null
    w = w - w_null @ scaling_factor # sidelobe-canceler equation

# Plot beam pattern
N_fft = 1024
w_padded = np.concatenate((w.squeeze(), np.zeros(N_fft - Nr))) # zero pad to N_fft elements to get more resolution in the FFT
w_fft_dB = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(w_padded)))**2) # magnitude of fft in dB
w_fft_dB -= np.max(w_fft_dB) # normalize to 0 dB at peak
theta_bins = np.arcsin(np.linspace(-1, 1, N_fft)) # Map the FFT bins to angles in radians

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta_bins, w_fft_dB)
# Add dots where nulls and SOI are
for null_rad in nulls_rad:
    ax.plot([null_rad], [0], 'or')
ax.plot([theta_soi], [0], 'og')
ax.set_theta_zero_location('N') # make 0 degrees point up
ax.set_theta_direction(-1) # increase clockwise
ax.set_thetagrids(np.arange(-90, 105, 15)) # it's in degrees
ax.set_rlabel_position(55) # Move grid labels away from other labels
ax.set_thetamin(-90) # only show top half
ax.set_thetamax(90)
ax.set_ylim([-40, 1]) # because there's no noise, only go down -40 dB
plt.show()
```


## MUSIC

```python
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

theta_scan = np.linspace(-1*np.pi, np.pi, 1000) # -180 to +180 degrees
results = []
for theta_i in theta_scan:
    s = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(theta_i)) # Steering Vector
    s = s.reshape(-1,1)
    metric = 1 / (s.conj().T @ V @ V.conj().T @ s) # The main MUSIC equation
    metric = np.abs(metric.squeeze()) # take magnitude
    metric = 10*np.log10(metric) # convert to dB
    results.append(metric)

results /= np.max(results) # normalize
```

```python
plot(10*np.log10(np.abs(w)),'.-')
```


## Root MUSIC

```python
num_expected_signals = 3

# Same eigendecomposition as MUSIC
R = np.cov(X)
w, v = np.linalg.eig(R)
eig_val_order = np.argsort(np.abs(w))
v = v[:, eig_val_order]
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
print("Estimated DOAs (degrees):", doas_deg)
```


## LMS

```python
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

w_lms /= np.linalg.norm(w_lms) # normalize weights

plt.plot(error_log)
plt.xlabel('Iteration')
plt.ylabel('Mean Square Error')
plt.show()

# Plot the beam pattern as shown previously
```


## Training Data

```python
import matplotlib.pyplot as plt
import numpy as np

# Array params
center_freq = 3.3e9
sample_rate = 30e6
d = 0.045 * center_freq / 3e8
print("d:", d)

# Includes all three signals, we'll call C our SOI
filename = '3p3G_A_B_C.npy'
X = np.load(filename)
Nr = X.shape[0]
```

```python
# Perform DOA to find angle of arrival of C
theta_scan = np.linspace(-1*np.pi/2, np.pi/2, 10000) # between -90 and +90 degrees
results = []
R = X @ X.conj().T # Calc covariance matrix. gives a Nr x Nr covariance matrix of the samples
Rinv = np.linalg.pinv(R) # pseudo-inverse tends to work better than a true inverse
for theta_i in theta_scan:
   a = np.exp(2j * np.pi * d * np.arange(X.shape[0]) * np.sin(theta_i)) # steering vector in the desired direction theta_i
   a = a.reshape(-1,1) # make into a column vector
   power = 1/(a.conj().T @ Rinv @ a).squeeze() # MVDR power equation
   power_dB = 10*np.log10(np.abs(power)) # power in signal, in dB so its easier to see small and large lobes at the same time
   results.append(power_dB)
results -= np.max(results) # normalize to 0 dB at peak
```

```python
# Pull out angle of C, after zeroing out the angles that include the interferers
results_temp = np.array(results)
results_temp[int(len(results)*0.4):] = -9999*np.ones(int(len(results)*0.6))
max_angle = theta_scan[np.argmax(results_temp)] # radians
print("max_angle:", max_angle)
```

```python
# Calc MVDR weights
s = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(max_angle)) # steering vector in the desired direction theta
s = s.reshape(-1,1) # make into a column vector
w = (Rinv @ s)/(s.conj().T @ Rinv @ s) # MVDR/Capon equation
```

```python
# Calc beam pattern
w = w.squeeze()
N_fft = 2048
w_padded = np.concatenate((w, np.zeros(N_fft - Nr))) # zero pad to N_fft elements to get more resolution in the FFT
w_fft_dB = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(w_padded)))**2) # magnitude of fft in dB
w_fft_dB -= np.max(w_fft_dB) # normalize to 0 dB at peak
theta_bins = np.arcsin(np.linspace(-1, 1, N_fft)) # Map the FFT bins to angles in radians

# Plot beam pattern and DOA results
plt.plot(theta_bins * 180 / np.pi, w_fft_dB) # MAKE SURE TO USE RADIAN FOR POLAR
plt.plot(theta_scan * 180 / np.pi, results, 'r')
plt.vlines(ymax=np.max(results), ymin=np.min(results) , x=max_angle*180/np.pi, color='g', linestyle='--')
plt.xlabel("Angle [deg]")
plt.ylabel("Magnitude [dB]")
plt.title("Beam Pattern and DOA Results, Without Training")
plt.grid()
plt.show()
```

```python
# Load "training data" which is just A and B, then calc Rinv
filename = '3p3G_A_B.npy'
X_A_B = np.load(filename)
R_training = X_A_B @ X_A_B.conj().T # Calc covariance matrix
Rinv_training = np.linalg.pinv(R_training)
```

```python
# Calc MVDR weights using training Rinv
s = np.exp(2j * np.pi * d * np.arange(Nr) * np.sin(max_angle)) # steering vector in the desired direction theta
s = s.reshape(-1,1) # make into a column vector (size 3x1)
w = (Rinv_training @ s)/(s.conj().T @ Rinv_training @ s) # MVDR/Capon equation
```


## Simulating Wideband Interferers

```python
N = 10 # number of elements in ULA
num_samples = 10000
d = 0.5

num_jammers = 3
jammer_pow_dB = np.array([30, 30, 30]) # Jammer powers in dB
jammer_aoa_deg = np.array([-70, -20, 40])  # Jammer angles in degrees
jammer_aoa = np.sin(np.deg2rad(jammer_aoa_deg)) * np.pi
element_gain_dB = np.zeros(N) # Gains in dB for the array elements (all 0 dB in our case)
element_gain_linear = 10.0 ** (element_gain_dB / 10) # Convert array gains to linear numbers
fractional_bw = 0.1 # if this is 0, the method matches the traditional way of using array factor to simulate received signals

# Build NxN jammer covariance matrix R
R = np.zeros((N, N), dtype=complex)
for m in range(N):
    for n in range(N):
        for j in range(num_jammers):
            total_element_gain = np.sqrt(element_gain_linear[m] * element_gain_linear[n])
            sinc_term = np.sinc(0.5 * fractional_bw * (m - n) * jammer_aoa[j] / np.pi)
            exp_term = np.exp(1j * (m - n) * jammer_aoa[j])
            R[m, n] += 10.0 ** (jammer_pow_dB[j] / 10) * total_element_gain * sinc_term * exp_term
R = np.eye(N, dtype=complex) + R

# Generate received samples
A = fractional_matrix_power(R, 0.5) # Compute the matrix square-root (effective Cholesky factorization)
A = A / np.sqrt(2)
X = np.zeros((N, num_samples), dtype=complex)
for k in range(num_samples):
    noise_vec = np.random.randn(N) + 1j * np.random.randn(N) # complex noise
    X[:, k] = A.conj().T @ noise_vec
```


## Circular Arrays

```python
radius = 0.05 # normalized by wavelength!
d = np.sqrt(2 * radius**2 * (1 - np.cos(2*np.pi/Nr)))
sf = 1.0 / (np.sqrt(2.0) * np.sqrt(1.0 - np.cos(2*np.pi/Nr))) # scaling factor based on geometry, eg for a hexagon it is 1.0
x = d * sf * np.cos(2 * np.pi / Nr * np.arange(Nr))
y = -1 * d * sf * np.sin(2 * np.pi / Nr * np.arange(Nr))
s = np.exp(1j * 2 * np.pi * (x * np.cos(theta) + y * np.sin(theta)))
s = s.reshape(-1, 1) # Nrx1
```
