"""Plot the simulation geometry: room, microphone array, and swept sources.

Reuses the exact constants from run_sim.py so the figure always matches the
simulation. Output: plots/room_geometry.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from run_sim import ANGLES_DEG, ARRAY_CENTER, MIC_DISTANCE, PLOTS_DIR, ROOM_DIMS, SOURCE_RADIUS

# Colors: validated categorical slots (mics=blue, sources=green) + ink tokens.
COL_MIC = "#2a78d6"
COL_SOURCE = "#008300"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"


def main():
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # Room outline
    ax.add_patch(
        Rectangle((0, 0), ROOM_DIMS[0], ROOM_DIMS[1],
                  fill=False, edgecolor=INK_SECONDARY, linewidth=1.5)
    )

    # Source positions along the swept arc
    thetas = np.radians(ANGLES_DEG)
    src_x = ARRAY_CENTER[0] + SOURCE_RADIUS * np.sin(thetas)
    src_y = ARRAY_CENTER[1] + SOURCE_RADIUS * np.cos(thetas)
    arc = np.radians(np.linspace(-80, 80, 200))
    ax.plot(ARRAY_CENTER[0] + SOURCE_RADIUS * np.sin(arc),
            ARRAY_CENTER[1] + SOURCE_RADIUS * np.cos(arc),
            linestyle="--", color=BASELINE, linewidth=1, zorder=1)
    ax.scatter(src_x, src_y, s=45, color=COL_SOURCE, zorder=3,
               label="source positions (2.0 m arc)")

    # Selective angle labels, placed just outside the arc
    for angle in (-80, -40, 0, 40, 80):
        t = np.radians(angle)
        lx = ARRAY_CENTER[0] + (SOURCE_RADIUS + 0.28) * np.sin(t)
        ly = ARRAY_CENTER[1] + (SOURCE_RADIUS + 0.28) * np.cos(t)
        ax.text(lx, ly, f"{angle:+d}°" if angle else "0°",
                ha="center", va="center", fontsize=9, color=INK_SECONDARY)

    # Broadside direction (0 deg reference)
    ax.annotate(
        "", xy=(ARRAY_CENTER[0], ARRAY_CENTER[1] + 1.1),
        xytext=(ARRAY_CENTER[0], ARRAY_CENTER[1]),
        arrowprops=dict(arrowstyle="->", color=INK_MUTED, linewidth=1.2),
    )
    ax.text(ARRAY_CENTER[0] + 0.08, ARRAY_CENTER[1] + 0.75, "broadside",
            fontsize=9, color=INK_MUTED, rotation=90, va="center")

    # Microphones (nearly coincident at room scale — see inset)
    mic_x = [ARRAY_CENTER[0] - MIC_DISTANCE / 2, ARRAY_CENTER[0] + MIC_DISTANCE / 2]
    mic_y = [ARRAY_CENTER[1], ARRAY_CENTER[1]]
    ax.scatter(mic_x, mic_y, s=70, marker="^", color=COL_MIC, zorder=4,
               label="microphones (0.15 m apart)")

    # Inset: zoom on the mic array so the spacing is visible
    axins = ax.inset_axes([0.66, 0.05, 0.30, 0.22])
    axins.set_facecolor(SURFACE)
    axins.scatter(mic_x, mic_y, s=90, marker="^", color=COL_MIC, zorder=3)
    for x, name in zip(mic_x, ("mic 0", "mic 1")):
        axins.text(x, ARRAY_CENTER[1] - 0.045, name, ha="center", fontsize=8,
                   color=INK_SECONDARY)
    axins.annotate(
        "", xy=(mic_x[1], ARRAY_CENTER[1] + 0.05),
        xytext=(mic_x[0], ARRAY_CENTER[1] + 0.05),
        arrowprops=dict(arrowstyle="<->", color=INK_MUTED, linewidth=1),
    )
    axins.text(ARRAY_CENTER[0], ARRAY_CENTER[1] + 0.075, f"{MIC_DISTANCE} m",
               ha="center", fontsize=8, color=INK_SECONDARY)
    axins.set_xlim(ARRAY_CENTER[0] - 0.22, ARRAY_CENTER[0] + 0.22)
    axins.set_ylim(ARRAY_CENTER[1] - 0.12, ARRAY_CENTER[1] + 0.14)
    axins.set_xticks([])
    axins.set_yticks([])
    for spine in axins.spines.values():
        spine.set_color(BASELINE)
    ax.indicate_inset_zoom(axins, edgecolor=BASELINE)

    ax.set_xlim(-0.4, ROOM_DIMS[0] + 0.4)
    ax.set_ylim(-0.4, ROOM_DIMS[1] + 0.4)
    ax.set_aspect("equal")
    ax.grid(color=GRIDLINE, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(BASELINE)
    ax.set_xlabel("x (m)", color=INK_SECONDARY)
    ax.set_ylabel("y (m)", color=INK_SECONDARY)
    ax.set_title(
        f"Simulation geometry: {ROOM_DIMS[0]:.0f} × {ROOM_DIMS[1]:.0f} m room, "
        f"2-mic array, sources at {ANGLES_DEG[0]}°…{ANGLES_DEG[-1]}°",
        color=INK_PRIMARY, fontsize=11,
    )
    leg = ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    for text in leg.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "room_geometry.png")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
