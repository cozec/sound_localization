"""Verify the GCC-PHAT -> TDOA -> angle pipeline against pyroomacoustics.

Simulates a 2-mic array in a room, places a noise source at known angles,
runs the pipeline on the simulated microphone signals, and compares the
estimated direction of arrival to ground truth.

Scenarios:
    anechoic     -- no reflections, no sensor noise (best case)
    reverberant  -- RT60 = 0.3 s shoebox room + sensor noise at 15 dB SNR

Outputs:
    results/sim_results.csv
    plots/angle_sweep.png
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra

from gcc_phat import SPEED_OF_SOUND, estimate_angle

FS = 16000
MIC_DISTANCE = 0.15   # meters between the two microphones
SOURCE_RADIUS = 2.0   # meters from array center (far field vs 0.15 m spacing)
ROOM_DIMS = [6.0, 5.0]
ARRAY_CENTER = np.array([3.0, 2.0])
ANGLES_DEG = np.arange(-80, 81, 10)
SIGNAL_DURATION = 0.5  # seconds of white noise

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")


def build_room(scenario, rng):
    """Create a 2-D shoebox room for the given scenario name."""
    if scenario == "anechoic":
        room = pra.ShoeBox(ROOM_DIMS, fs=FS, max_order=0)
    elif scenario == "reverberant":
        e_absorption, max_order = pra.inverse_sabine(0.3, ROOM_DIMS)
        room = pra.ShoeBox(
            ROOM_DIMS,
            fs=FS,
            materials=pra.Material(e_absorption),
            max_order=max_order,
        )
    else:
        raise ValueError(f"unknown scenario: {scenario}")
    return room


def simulate_capture(true_angle_deg, scenario, rng):
    """Simulate one noise burst and return the two microphone signals.

    The array lies on the x-axis; broadside (0 degrees) points to +y.
    Positive angles are toward the +x microphone (mic index 1).
    """
    room = build_room(scenario, rng)

    mic_positions = np.array(
        [
            [ARRAY_CENTER[0] - MIC_DISTANCE / 2, ARRAY_CENTER[0] + MIC_DISTANCE / 2],
            [ARRAY_CENTER[1], ARRAY_CENTER[1]],
        ]
    )
    room.add_microphone_array(pra.MicrophoneArray(mic_positions, fs=FS))

    theta = np.radians(true_angle_deg)
    source_pos = ARRAY_CENTER + SOURCE_RADIUS * np.array([np.sin(theta), np.cos(theta)])
    signal = rng.standard_normal(int(SIGNAL_DURATION * FS))
    room.add_source(source_pos.tolist(), signal=signal)

    if scenario == "reverberant":
        room.simulate(reference_mic=0, snr=15)
    else:
        room.simulate()

    return room.mic_array.signals  # shape (2, n_samples)


def run_sweep(scenario, rng):
    """Run the angle sweep for one scenario; return list of result rows."""
    rows = []
    for angle in ANGLES_DEG:
        signals = simulate_capture(angle, scenario, rng)
        # sig = -x mic (index 0), refsig = +x mic (index 1): a source toward
        # +x reaches mic 1 first, mic 0 lags -> tau > 0 -> positive angle.
        est = estimate_angle(signals[0], signals[1], FS, MIC_DISTANCE)
        rows.append(
            {
                "scenario": scenario,
                "true_angle_deg": float(angle),
                "estimated_angle_deg": round(est, 2),
                "error_deg": round(est - float(angle), 2),
            }
        )
    return rows


def main():
    rng = np.random.default_rng(42)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    all_rows = []
    for scenario in ("anechoic", "reverberant"):
        rows = run_sweep(scenario, rng)
        errors = np.array([r["error_deg"] for r in rows])
        print(
            f"{scenario:12s}  mean |error| = {np.abs(errors).mean():5.2f} deg   "
            f"max |error| = {np.abs(errors).max():5.2f} deg"
        )
        all_rows.extend(rows)

    csv_path = os.path.join(RESULTS_DIR, "sim_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"wrote {csv_path}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for scenario, marker in (("anechoic", "o"), ("reverberant", "s")):
        rows = [r for r in all_rows if r["scenario"] == scenario]
        true_a = [r["true_angle_deg"] for r in rows]
        est_a = [r["estimated_angle_deg"] for r in rows]
        err = [r["error_deg"] for r in rows]
        ax1.plot(true_a, est_a, marker, label=scenario, alpha=0.8)
        ax2.plot(true_a, err, marker + "-", label=scenario, alpha=0.8)

    lim = [-90, 90]
    ax1.plot(lim, lim, "k--", linewidth=0.8, label="ideal")
    ax1.set_xlabel("true angle (deg)")
    ax1.set_ylabel("estimated angle (deg)")
    ax1.set_title("GCC-PHAT DOA: estimated vs true")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("true angle (deg)")
    ax2.set_ylabel("error (deg)")
    ax2.set_title("Estimation error")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "angle_sweep.png")
    fig.savefig(plot_path, dpi=150)
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
