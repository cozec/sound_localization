"""GCC-PHAT time-difference-of-arrival estimation and angle conversion.

Implements the first two stages of the pipeline:
    GCC-PHAT cross-correlation -> TDOA -> direction-of-arrival angle.
"""

import numpy as np

SPEED_OF_SOUND = 343.0  # m/s at ~20 C


def gcc_phat(sig, refsig, fs, max_tau=None, interp=16):
    """Estimate the time delay of `sig` relative to `refsig` using GCC-PHAT.

    Parameters
    ----------
    sig : np.ndarray
        Signal at microphone 1 (1-D).
    refsig : np.ndarray
        Signal at microphone 2 (1-D), used as the reference.
    fs : int
        Sample rate in Hz.
    max_tau : float, optional
        Physical bound on the delay in seconds (e.g. mic_distance / c).
        Correlation peaks outside +/- max_tau are ignored.
    interp : int
        Upsampling factor of the inverse FFT for sub-sample precision.

    Returns
    -------
    tau : float
        Estimated delay in seconds. Positive means `sig` lags `refsig`.
    cc : np.ndarray
        The (upsampled) cross-correlation slice searched for the peak.
    """
    n = len(sig) + len(refsig)

    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)
    # Phase transform: whiten the magnitude, keep only phase information.
    R /= np.abs(R) + 1e-15

    cc = np.fft.irfft(R, n=interp * n)

    max_shift = interp * n // 2
    if max_tau is not None:
        max_shift = min(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    shift = np.argmax(np.abs(cc)) - max_shift
    tau = shift / float(interp * fs)
    return tau, cc


def tdoa_to_angle(tau, mic_distance, c=SPEED_OF_SOUND):
    """Convert a TDOA to a far-field direction-of-arrival angle.

    Parameters
    ----------
    tau : float
        Time difference of arrival in seconds (mic 1 minus mic 2).
    mic_distance : float
        Spacing between the two microphones in meters.
    c : float
        Speed of sound in m/s.

    Returns
    -------
    float
        Angle in degrees from broadside (0 = directly in front,
        +90 = toward mic 1, -90 = toward mic 2). A two-mic array cannot
        resolve front/back, so the range is [-90, +90].
    """
    sin_theta = np.clip(c * tau / mic_distance, -1.0, 1.0)
    return float(np.degrees(np.arcsin(sin_theta)))


def estimate_angle(sig, refsig, fs, mic_distance, c=SPEED_OF_SOUND, interp=16):
    """Full pipeline for one frame: GCC-PHAT -> TDOA -> angle in degrees."""
    max_tau = mic_distance / c
    tau, _ = gcc_phat(sig, refsig, fs, max_tau=max_tau, interp=interp)
    return tdoa_to_angle(tau, mic_distance, c=c)
