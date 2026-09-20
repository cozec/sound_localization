"""PySDR "Circular Arrays" section as a standalone demo.

Steering-vector code from https://pysdr.org/content/doa.html (Marc Lichtman,
PySDR, CC BY-NC-SA 4.0). The page gives only the UCA steering vector and
says "all the code so far applies, scan 0..360 instead of -90..90"; this
script does that: 5-element UCA (KrakenSDR-like), two sources, and the
conventional / MVDR / MUSIC scans over the full circle, next to a 5-element
ULA on the same scene to show the ULA's front/back ambiguity.

UCA steering vector (positions in wavelengths, angle measured from +x,
counter-clockwise):
    x_k = r cos(2πk/Nr),  y_k = -r sin(2πk/Nr)
    s_k(θ) = exp(j 2π (x_k cosθ + y_k sinθ))
The page's d*sf product simplifies to r exactly; it is kept as written.

Output: plots/pysdr_uca.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")

np.random.seed(0)

sample_rate = 1e6
N = 10000
t = np.arange(N)/sample_rate
Nr = 5
SOURCES = [(30, 0.01e6, 1.0), (150, 0.02e6, 1.0)]   # (deg, tone Hz, amplitude); 150 = 180-30, mirror pair for a ULA
NOISE = 0.1
K = len(SOURCES)


def uca_positions(radius):
    """Element x, y in wavelengths (page code)."""
    d = np.sqrt(2 * radius**2 * (1 - np.cos(2*np.pi/Nr)))
    sf = 1.0 / (np.sqrt(2.0) * np.sqrt(1.0 - np.cos(2*np.pi/Nr))) # scaling factor based on geometry, eg for a hexagon it is 1.0
    x = d * sf * np.cos(2 * np.pi / Nr * np.arange(Nr))
    y = -1 * d * sf * np.sin(2 * np.pi / Nr * np.arange(Nr))
    return x, y


def steer_uca(theta, radius):
    x, y = uca_positions(radius)
    s = np.exp(1j * 2 * np.pi * (x * np.cos(theta) + y * np.sin(theta)))
    return s.reshape(-1, 1) # Nrx1


def steer_ula(theta, d=0.5):
    """ULA along the x axis, same angle convention (theta from +x): phase = 2π d k cosθ."""
    return np.exp(2j * np.pi * d * np.arange(Nr) * np.cos(theta)).reshape(-1, 1)


def simulate(steer):
    X = np.zeros((Nr, N), dtype=complex)
    for deg, f, amp in SOURCES:
        X += amp * steer(np.radians(deg)) @ np.exp(2j*np.pi*f*t).reshape(1, -1)
    return X + NOISE * (np.random.randn(Nr, N) + 1j*np.random.randn(Nr, N))


def scans(X, steer, theta_scan):
    R = (X @ X.conj().T) / N
    Rinv = np.linalg.pinv(R)
    w, v = np.linalg.eig(R)
    v = v[:, np.argsort(np.abs(w))]
    V = v[:, :Nr - K]                         # noise subspace
    conv, mvdr, music = [], [], []
    for th in theta_scan:
        s = steer(th)
        conv.append(10*np.log10(np.var((s / Nr).conj().T @ X)))
        mvdr.append(10*np.log10(np.abs(1 / (s.conj().T @ Rinv @ s).squeeze())))
        music.append(10*np.log10(np.abs(1 / (s.conj().T @ V @ V.conj().T @ s).squeeze())))
    return [np.array(a) - np.max(a) for a in (conv, mvdr, music)]


theta_scan = np.linspace(0, 2*np.pi, 1441)   # 0 to 360 degrees
RADIUS = 0.425                                # adjacent spacing 2 r sin(pi/5) = 0.5 wavelengths
uca = lambda th: steer_uca(th, RADIUS)

X_uca = simulate(uca)
X_ula = simulate(steer_ula)
conv_u, mvdr_u, music_u = scans(X_uca, uca, theta_scan)
conv_l, mvdr_l, music_l = scans(X_ula, steer_ula, theta_scan)

from scipy.signal import find_peaks
deg = np.degrees(theta_scan)
print(f"UCA radius {RADIUS} λ, adjacent spacing {2*RADIUS*np.sin(np.pi/Nr):.3f} λ; sources at {[s[0] for s in SOURCES]}°")
for name, sc in (("UCA conventional", conv_u), ("UCA MVDR", mvdr_u), ("UCA MUSIC", music_u),
                 ("ULA conventional", conv_l), ("ULA MVDR", mvdr_l), ("ULA MUSIC", music_l)):
    pk, _ = find_peaks(sc[:-1], prominence=3)
    print(f"  {name:17s} peaks: {[round(deg[i], 1) for i in pk]}")

# Conventional beamwidth vs radius (the page's radius = 0.05 is nearly a point)
theta_pat = np.linspace(-np.pi, np.pi, 1441)


def uca_pattern(radius, look_deg=30):
    w = steer_uca(np.radians(look_deg), radius) / Nr
    p = np.abs(np.hstack([w.conj().T @ steer_uca(th, radius) for th in theta_pat]).squeeze())
    return 20*np.log10(p + 1e-12)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 10))


def style(ax, title):
    ax.set_theta_zero_location("E"); ax.set_theta_direction(1)   # angle from +x, counter-clockwise (page convention)
    ax.set_rlim(-40, 0); ax.set_rlabel_position(255)
    ax.set_title(title, pad=14)
    for deg_, _, _ in SOURCES:
        ax.plot([np.radians(deg_)] * 2, [-40, 0], "k:", lw=0.8)


ax = fig.add_subplot(2, 2, 1)
x, y = uca_positions(RADIUS)
ax.plot(x, y, "o", ms=9, label=f"UCA, r = {RADIUS} λ")
for k in range(Nr):
    ax.annotate(str(k), (x[k], y[k]), textcoords="offset points", xytext=(6, 6))
ax.plot(0.5*np.arange(Nr) - 1.0, np.zeros(Nr) - 0.7, "s", ms=7, color="C3", label="ULA, d = 0.5 λ")
for deg_, _, _ in SOURCES:
    ax.annotate("", xy=(0.9*np.cos(np.radians(deg_)), 0.9*np.sin(np.radians(deg_))), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="k", lw=1))
    ax.text(1.0*np.cos(np.radians(deg_)), 1.0*np.sin(np.radians(deg_)), f"{deg_}°", ha="center")
ax.set_aspect("equal"); ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.1, 1.2)
ax.set_xlabel("x [λ]"); ax.set_ylabel("y [λ]"); ax.grid(alpha=0.3); ax.legend(loc="lower right")
ax.set_title("Geometry (angle from +x, CCW)")

ax = fig.add_subplot(2, 2, 2, projection="polar")
ax.plot(theta_scan, np.maximum(conv_u, -40), label="conventional", lw=1)
ax.plot(theta_scan, np.maximum(mvdr_u, -40), label="MVDR", lw=1)
ax.plot(theta_scan, np.maximum(music_u, -40), label="MUSIC (K=2)", lw=1.5)
style(ax, "UCA: full 360° scan — two peaks, no ambiguity")
ax.legend(loc="lower left", bbox_to_anchor=(-0.15, -0.1), fontsize=8)

ax = fig.add_subplot(2, 2, 3, projection="polar")
ax.plot(theta_scan, np.maximum(conv_l, -40), label="conventional", lw=1)
ax.plot(theta_scan, np.maximum(mvdr_l, -40), label="MVDR", lw=1)
ax.plot(theta_scan, np.maximum(music_l, -40), label="MUSIC (K=2)", lw=1.5)
style(ax, "ULA on the same scene — mirror peaks about the array axis")
ax.legend(loc="lower left", bbox_to_anchor=(-0.15, -0.1), fontsize=8)

ax = fig.add_subplot(2, 2, 4, projection="polar")
for r, c in ((0.05, "C3"), (0.2, "C1"), (RADIUS, "C0"), (1.0, "C2")):
    ax.plot(theta_pat, np.maximum(uca_pattern(r), -40), color=c, lw=1.2, label=f"r = {r} λ")
ax.set_theta_zero_location("E"); ax.set_theta_direction(1); ax.set_rlim(-40, 0); ax.set_rlabel_position(255)
ax.set_title("UCA conventional beam steered to 30° vs radius\n(page's r = 0.05 λ is almost omnidirectional)", pad=14)
ax.legend(loc="lower left", bbox_to_anchor=(-0.15, -0.1), fontsize=8)

fig.tight_layout()
os.makedirs(PLOTS_DIR, exist_ok=True)
out = os.path.join(PLOTS_DIR, "pysdr_uca.png")
fig.savefig(out, dpi=130)
print(f"wrote {out}")
