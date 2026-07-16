"""Delay-and-sum beamformer for a 2-microphone array.

For each candidate steering angle, one channel is fractionally delayed
(via an FFT phase shift) so a source at that angle would align coherently,
the channels are summed, and the output power is recorded. The steering
angle with maximum power is the direction-of-arrival estimate.
"""

import numpy as np

from gcc_phat import SPEED_OF_SOUND


def steered_power(signals, fs, mic_distance, angles_deg, c=SPEED_OF_SOUND):
    """Compute delay-and-sum output power over a set of steering angles.

    Parameters
    ----------
    signals : np.ndarray
        Shape (2, n_samples); row 0 is the -x mic, row 1 the +x mic
        (same convention as run_sim: positive angles toward the +x mic).
    fs : int
        Sample rate in Hz.
    mic_distance : float
        Mic spacing in meters.
    angles_deg : array-like
        Candidate steering angles in degrees from broadside.
    c : float
        Speed of sound in m/s.

    Returns
    -------
    np.ndarray
        Mean output power of the summed beam at each angle.
    """
    x0 = np.asarray(signals[0], dtype=float)
    x1 = np.asarray(signals[1], dtype=float)
    n = len(x0)
    X1 = np.fft.rfft(x1)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)

    powers = np.empty(len(angles_deg))
    for i, ang in enumerate(np.asarray(angles_deg, dtype=float)):
        # For a source at +ang, mic 0 receives the wave tau seconds after
        # mic 1; delaying mic 1 by tau aligns the two channels.
        tau = mic_distance * np.sin(np.radians(ang)) / c
        x1_delayed = np.fft.irfft(X1 * np.exp(-2j * np.pi * freqs * tau), n)
        beam = x0 + x1_delayed
        powers[i] = np.mean(beam**2)
    return powers


def estimate_angle_das(signals, fs, mic_distance, resolution_deg=0.5,
                       c=SPEED_OF_SOUND):
    """DOA estimate: steering angle that maximizes delay-and-sum power."""
    angles = np.arange(-90, 90 + resolution_deg, resolution_deg)
    powers = steered_power(signals, fs, mic_distance, angles, c=c)
    return float(angles[np.argmax(powers)]), angles, powers
