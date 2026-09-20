"""How conventional beamsteering works on a uniform circular array.

Four panels: (1) wavefronts and each element's path difference for a wave
from the look direction; (2) per-element phase lead vs arrival angle (a
shifted cosine per element); (3) the phasors exp(j[phi_k(theta) -
phi_k(look)]) summed tip-to-tail for a wave from the look direction and from
an off-look angle; (4) the resulting beam pattern |w^H s(theta)|.

Output: plots/uca_explain.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")

Nr, r = 5, 0.425
LOOK = 30.0
OFF = 100.0
phi_k = 2*np.pi*np.arange(Nr)/Nr
x, y = r*np.cos(phi_k), -r*np.sin(phi_k)
cols = [f"C{k}" for k in range(Nr)]

def phase(theta_deg):                      # per-element phase lead, radians
    th = np.radians(theta_deg)
    return 2*np.pi*(x*np.cos(th) + y*np.sin(th))

fig = plt.figure(figsize=(15, 10))

# ---------- (1) geometry: wavefronts and path differences ----------
ax = fig.add_subplot(2, 2, 1)
th = np.radians(LOOK); u = np.array([np.cos(th), np.sin(th)]); n = np.array([-np.sin(th), np.cos(th)])
for c in np.arange(-1.0, 1.01, 0.25):       # wavefronts (lines perpendicular to u), spaced 0.25 wavelength
    p0 = c*u
    ax.plot([p0[0]-1.2*n[0], p0[0]+1.2*n[0]], [p0[1]-1.2*n[1], p0[1]+1.2*n[1]], color="0.85", lw=0.8)
ax.annotate("", xy=(0.95*u[0], 0.95*u[1]), xytext=(1.3*u[0], 1.3*u[1]),
            arrowprops=dict(arrowstyle="->", lw=2))
ax.text(1.32*u[0], 1.32*u[1]+0.05, f"wave from θ = {LOOK:.0f}°", ha="center")
for k in range(Nr):
    proj = (x[k]*u[0] + y[k]*u[1])          # projection onto u, in wavelengths
    ax.plot([x[k], proj*u[0]], [y[k], proj*u[1]], color=cols[k], ls="--", lw=1)
    ax.plot(x[k], y[k], "o", color=cols[k], ms=10)
    ax.annotate(f"{k}: {proj:+.3f} λ", (x[k], y[k]), textcoords="offset points", xytext=(8, 6), color=cols[k], fontsize=9)
ax.plot([-1.2*u[0], 1.2*u[0]], [-1.2*u[1], 1.2*u[1]], "k:", lw=0.8)
ax.plot(0, 0, "k+", ms=10)
ax.set_aspect("equal"); ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.2, 1.5)
ax.set_title("(1) Path difference = projection of element position onto the wave direction\nphase lead φ_k = 2π·(x_k cosθ + y_k sinθ)")
ax.set_xlabel("x [λ]"); ax.set_ylabel("y [λ]")

# ---------- (2) phase of each element vs arrival angle ----------
ax = fig.add_subplot(2, 2, 2)
ths = np.linspace(0, 360, 721)
P = np.array([phase(t) for t in ths])       # (721, Nr)
for k in range(Nr):
    ax.plot(ths, np.degrees(P[:, k]), color=cols[k], label=f"element {k}")
ax.axvline(LOOK, color="k", ls=":"); ax.axvline(OFF, color="r", ls=":")
ax.text(LOOK+2, 150, "look", fontsize=9); ax.text(OFF+2, 150, "off-look", color="r", fontsize=9)
ax.set_xlabel("arrival angle θ [deg]"); ax.set_ylabel("phase lead φ_k(θ) [deg]")
ax.set_title("(2) UCA: φ_k(θ) = 2πr·cos(θ + 2πk/Nr) — a shifted cosine per element\n(ULA would be 2πkd·sinθ: same curve scaled by k)")
ax.set_xlim(0, 360); ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=5, loc="lower center")

# ---------- (3) phasors after applying w = s(look)^H ----------
def phasor_panel(ax, theta_deg, title):
    resid = phase(theta_deg) - phase(LOOK)  # exponent of each term in w^H s(θ)
    z = np.exp(1j*resid)
    tip = 0
    for k in range(Nr):                     # tip-to-tail sum
        ax.annotate("", xy=(np.real(tip+z[k]), np.imag(tip+z[k])), xytext=(np.real(tip), np.imag(tip)),
                    arrowprops=dict(arrowstyle="->", color=cols[k], lw=2))
        tip += z[k]
    ax.annotate("", xy=(np.real(tip), np.imag(tip)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="k", lw=2.5, ls="-"))
    g = abs(tip)/Nr
    ax.set_title(f"{title}\n|sum|/Nr = {g:.2f}  ({20*np.log10(max(g,1e-3)):+.1f} dB)")
    ax.set_aspect("equal"); ax.set_xlim(-2.5, 5.5); ax.set_ylim(-3, 3); ax.grid(alpha=0.3)
    ax.axhline(0, color="0.7", lw=0.8); ax.axvline(0, color="0.7", lw=0.8)

ax = fig.add_subplot(2, 4, 5)
phasor_panel(ax, LOOK, f"(3a) wave from look direction {LOOK:.0f}°:\nall exp(j[φ_k(θ)−φ_k(look)]) = 1, add in line")
ax = fig.add_subplot(2, 4, 6)
phasor_panel(ax, OFF, f"(3b) wave from {OFF:.0f}°:\nphasors point different ways, partly cancel")

# ---------- (4) resulting beam pattern ----------
ax = fig.add_subplot(2, 2, 4, projection="polar")
gain = np.array([abs(np.sum(np.exp(1j*(phase(t) - phase(LOOK)))))/Nr for t in ths])
ax.plot(np.radians(ths), np.maximum(20*np.log10(gain+1e-9), -30), lw=1.8)
for t, c in ((LOOK, "k"), (OFF, "r")):
    ax.plot([np.radians(t)]*2, [-30, 0], color=c, ls=":")
ax.set_theta_zero_location("E"); ax.set_theta_direction(1); ax.set_rlim(-30, 0); ax.set_rlabel_position(255)
ax.set_title(f"(4) |wᴴs(θ)| over all θ = the beam pattern\n(black = look {LOOK:.0f}°, red = {OFF:.0f}° from panel 3b)", pad=14)

fig.tight_layout()
os.makedirs(PLOTS_DIR, exist_ok=True)
out = os.path.join(PLOTS_DIR, "uca_explain.png")
fig.savefig(out, dpi=120)
print(f"wrote {out}")
