"""MVDR (Capon) beamformer spatial spectrum for a 2-microphone array.

Broadband MVDR: STFT both channels, estimate a 2x2 spatial covariance per
frequency bin, and average the narrowband Capon pseudo-spectrum
P(theta) = 1 / (a^H R^-1 a) over the band. Unlike delay-and-sum, which
weights channels uniformly, MVDR minimizes output power subject to unit
gain toward the look direction, yielding a much sharper spatial peak.
"""

import numpy as np
from scipy.signal import stft

from gcc_phat import SPEED_OF_SOUND


def mvdr_spectrum(signals, fs, mic_distance, angles_deg, c=SPEED_OF_SOUND,
                  nperseg=256, band=(300.0, 7000.0), loading=1e-2):
    """Compute the broadband MVDR pseudo-spectrum over candidate angles.

    Parameters
    ----------
    signals : np.ndarray
        Shape (2, n_samples); row 0 is the -x mic, row 1 the +x mic
        (run_sim convention: positive angles toward the +x mic).
    fs : int
        Sample rate in Hz.
    mic_distance : float
        Mic spacing in meters.
    angles_deg : array-like
        Candidate look directions in degrees from broadside.
    c : float
        Speed of sound in m/s.
    nperseg : int
        STFT frame length.
    band : (float, float)
        Frequency band (Hz) whose bins are averaged.
    loading : float
        Diagonal loading factor (fraction of average channel power) for
        numerical robustness of the covariance inverse.

    Returns
    -------
    np.ndarray
        Pseudo-spectrum (linear power) at each candidate angle.
    """
    angles = np.asarray(angles_deg, dtype=float)
    freqs, _, X = stft(np.asarray(signals, dtype=float), fs=fs, nperseg=nperseg)
    # X shape: (2, n_bins, n_frames)
    bins = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]

    taus = mic_distance * np.sin(np.radians(angles)) / c
    spectrum = np.zeros(len(angles))
    eye = np.eye(2)

    for k in bins:
        Xk = X[:, k, :]  # (2, n_frames)
        R = (Xk @ Xk.conj().T) / Xk.shape[1]
        R = R + loading * (np.trace(R).real / 2) * eye
        Rinv = np.linalg.inv(R)
        # Steering vectors: mic 1 leads mic 0 by tau for a positive angle.
        A = np.vstack([np.ones_like(taus), np.exp(2j * np.pi * freqs[k] * taus)])
        denom = np.einsum("ia,ij,ja->a", A.conj(), Rinv, A).real
        spectrum += 1.0 / np.maximum(denom, 1e-12)

    return spectrum / len(bins)


def estimate_angle_mvdr(signals, fs, mic_distance, resolution_deg=0.5,
                        c=SPEED_OF_SOUND, **kwargs):
    """DOA estimate: look direction that maximizes the MVDR pseudo-spectrum."""
    angles = np.arange(-90, 90 + resolution_deg, resolution_deg)
    spectrum = mvdr_spectrum(signals, fs, mic_distance, angles, c=c, **kwargs)
    return float(angles[np.argmax(spectrum)]), angles, spectrum
