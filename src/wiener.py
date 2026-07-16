"""Multichannel Wiener filter (MWF) for a 2-microphone array.

Per STFT frequency bin, the MWF estimates the target signal component at a
reference microphone with minimum mean squared error:

    W(f) = (Rx(f) + delta*I)^-1 (Rx(f) - Rn(f)) e_ref

where Rx is the spatial covariance of the noisy mixture, Rn the covariance
of the noise alone (estimated from target-silent frames), and e_ref selects
the reference channel. Equivalently MVDR beamforming followed by a
single-channel Wiener postfilter.
"""

import numpy as np
from scipy.signal import istft, stft


def mwf_weights(X, noise_frames, loading=1e-2):
    """Estimate per-bin MWF weights from the mixture STFT.

    Parameters
    ----------
    X : np.ndarray
        Mixture STFT, shape (n_channels, n_bins, n_frames).
    noise_frames : np.ndarray
        Boolean mask over frames that contain noise only (target silent).
    loading : float
        Diagonal loading factor (fraction of average channel power).

    Returns
    -------
    np.ndarray
        Weights, shape (n_channels, n_bins); output is Y = W^H X per bin.
    """
    n_ch, n_bins, _ = X.shape
    eye = np.eye(n_ch)
    e_ref = np.zeros(n_ch)
    e_ref[0] = 1.0

    W = np.empty((n_ch, n_bins), dtype=complex)
    for k in range(n_bins):
        Xk = X[:, k, :]
        Nk = Xk[:, noise_frames]
        Rx = (Xk @ Xk.conj().T) / Xk.shape[1]
        Rn = (Nk @ Nk.conj().T) / max(Nk.shape[1], 1)
        Rs = Rx - Rn  # target covariance estimate
        Rx_loaded = Rx + loading * (np.trace(Rx).real / n_ch) * eye
        W[:, k] = np.linalg.solve(Rx_loaded, Rs @ e_ref)
    return W


def apply_weights(W, X):
    """Apply per-bin weights: Y[k, t] = W[:, k]^H X[:, k, t]."""
    return np.einsum("ck,ckt->kt", W.conj(), X)


def enhance(signals, fs, noise_seconds, nperseg=256, loading=1e-2):
    """Run the full MWF on a multichannel capture.

    Parameters
    ----------
    signals : np.ndarray
        Shape (n_channels, n_samples); channel 0 is the reference.
    fs : int
        Sample rate in Hz.
    noise_seconds : float
        Duration of the leading noise-only segment (target silent) used to
        estimate the noise covariance.
    nperseg : int
        STFT frame length.
    loading : float
        Diagonal loading factor for the covariance inverse.

    Returns
    -------
    enhanced : np.ndarray
        Time-domain MWF output (target estimate at the reference mic).
    W : np.ndarray
        The per-bin weights, shape (n_channels, n_bins).
    """
    freqs, times, X = stft(np.asarray(signals, dtype=float), fs=fs,
                           nperseg=nperseg)
    noise_frames = times < noise_seconds
    W = mwf_weights(X, noise_frames, loading=loading)
    Y = apply_weights(W, X)
    _, enhanced = istft(Y, fs=fs, nperseg=nperseg)
    return enhanced[: signals.shape[1]], W
