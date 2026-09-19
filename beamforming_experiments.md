# Beamforming experiments: from DOA to signal extraction

Notes from working through *what to do after the DOA is known* — how to use
the angle to pull a cleaner signal out of the array. The experiments follow
the [PySDR "Beamforming & DOA" chapter](https://pysdr.org/content/doa.html)
(Marc Lichtman, CC BY-NC-SA 4.0), whose code is pulled verbatim into
[reference/pysdr_doa_snippets.md](reference/pysdr_doa_snippets.md) and
stitched into runnable scripts under `src/pysdr_*.py`. PySDR is written for
antenna arrays; the math is identical for a microphone array with the
substitution `d_wavelengths = spacing · f / c`, applied per STFT bin.

All scripts run with `.venv/bin/python src/<name>.py` and write to `plots/`.

## 0. The one idea

A plane wave from angle θ reaches element `k` of a uniform linear array with
phase `2π·k·d·sinθ`. Stacking those phases gives the **steering vector**
`s(θ) = exp(2jπ·d·arange(Nr)·sinθ)`. Every method below is a choice of a
weight vector `w`, applied as `y = wᴴ x`:

| Method | Weights | What it does |
|---|---|---|
| Conventional / delay-and-sum | `w = s(θ)/Nr` | undo each element's phase lag (the conjugate), sum |
| MVDR / Capon | `w = R⁻¹s / (sᴴR⁻¹s)` | unit gain at θ, minimum output power → implicit nulls on whatever else is in `R` |
| LCMV | `w = R⁻¹C (CᴴR⁻¹C)⁻¹ f` | MVDR with several constraints: `C = [s₁ … s_K]`, gains `f` |
| Zero-forcing | `W = C (CᴴC)⁻¹` | least-squares unmixing, no covariance needed |

Spectral analysis and spatial filtering are the same math with time ↔ space
and frequency ↔ `d·sinθ`: the conventional scan is a spatial periodogram,
the beam pattern is literally `fft(w)` mapped through `arcsin`, MVDR/MUSIC
are Capon/Pisarenko spectral estimators applied across the array.

## 1. Polar beampatterns of our 2-mic array

`src/plot_polar.py` — closed-form delay-and-sum patterns steered to +40° at
500 Hz–4 kHz, and MVDR patterns with a simulated interferer at −30°.

![Polar beampatterns](plots/polar_patterns.png)

- Beam narrows with frequency; grating lobes above the aliasing limit
  `c/(2d) = 1143 Hz` for `d = 15 cm`; every pattern is mirror-symmetric about
  the mic axis (front/back ambiguity of any linear array).
- MVDR holds 0 dB at +40° and puts a >30 dB null at −30°. With two mics there
  is exactly one null to spend, and the DOA tells you where.
- At 2 kHz the +40° and −30° steering vectors are identical (one full cycle of
  inter-mic delay apart), so no null is possible — a concrete aliasing case.

## 2. PySDR chapter demo

`src/pysdr_doa_demo.py` — the chapter's core, end to end.

![Received signal](plots/pysdr_received_signal.png)

Three elements receive the same tone with a fixed phase offset between them
— that offset *is* the steering vector.

![PySDR demo](plots/pysdr_doa_demo.png)

Top: conventional scan (cartesian and polar) for one source at 20°, and the
FFT beam pattern. Bottom: 8 elements, sources at 20°, 25° and −40° (−20 dB).
Conventional merges 20°/25° into one blob; MVDR resolves them; MUSIC gives
needle peaks; Root MUSIC returns `[-40.03, 19.99, 25.01]` in closed form.

Observation on noise: without noise the scan's nulls go to −∞ and the main
lobe is unchanged. Noise only fills the nulls up to the noise floor
(~−22 dB at the chapter's noise level); it doesn't move the peak.

## 3. MVDR vs conventional, one source

`src/pysdr_mvdr.py` — the "MVDR/Capon Beamformer" section as a script.

![MVDR scans](plots/pysdr_mvdr.png)

![MVDR waveforms](plots/pysdr_mvdr_waveforms.png)

Weights for Part A (3 elements, source at 20°):

```
w_conv = [ 0.3333+0.0000j   0.1587+0.2931j  -0.1822+0.2792j ]   |w| = 0.333 each, phase 0 / 61.6° / 123.1°
w_mvdr = [ 0.2413-0.0372j   0.1471+0.3185j  -0.2358+0.3341j ]   |w| = 0.244 / 0.351 / 0.409
```

SNR: element 17.0 dB → conventional 21.8 dB → MVDR 21.6 dB. With a single
source in white noise MVDR has nothing to null and collapses to
delay-and-sum; both get the `10·log10(3) = 4.8 dB` array gain. MVDR's
magnitude tilt is finite-sample noise fitting (it costs 0.2 dB), which is
what diagonal loading fixes. MVDR's advantage is in the *scan* (sharper
peak), not in this output.

## 4. Two sources: the null appears

`src/pysdr_mvdr_two_sources.py` — target at +20°, equal-power interferer at
−40°, 3 elements.

![Two sources](plots/pysdr_mvdr_two_sources.png)

```
                 gain @ +20°     gain @ −40°     output SNR
element 0                                         −0.1 dB
conventional     1.000 (0 dB)    0.333 (−9.6 dB)   9.3 dB
MVDR             1.000 (0 dB)    0.003 (−51.8 dB) 21.1 dB
```

Now the weights differ for a reason: MVDR reshapes magnitudes to
0.17 / 0.50 / 0.33 so that `wᴴs(−40°) ≈ 0`. Conventional can't; the
interferer sits on a −9.6 dB sidelobe and visibly distorts the output.

## 5. Part B: detect three sources, then extract them

`src/pysdr_mvdr_part_b.py` — 8 elements, sources at +20°, +25° (equal) and
−40° (−20 dB). The chapter's stated goal is **detection**: three peaks a
peak-finder can extract.

![Part B](plots/pysdr_mvdr_part_b.png)

`scipy.signal.find_peaks(prominence=3 dB)` on the two scans:

| | peaks found |
|---|---|
| conventional | 22.5° (merged blob) + two sidelobes at −47°, −14° — the −40° source is not a peak |
| MVDR | **20.1°, 24.9°, −39.9°** — all three, within 0.1° |

Steered to +20° for extraction, MVDR nulls the +25° source at −58 dB (5°
off-look, inside the conventional beamwidth) and the output SNR goes from
1.5 dB to 26.2 dB. The price: MVDR's sidelobes elsewhere are *higher* than
conventional's — it minimizes power for this `R`, not sidelobes in general.

### Extracting all three at once

`src/pysdr_extract_three.py` — one weight vector per detected peak,
`Y = Wᴴ X` gives a 3 × N matrix of separated streams.

![Extract three](plots/pysdr_extract_three.png)

```
                 +20°    +25°    −40°     (output SNR, dB; inputs were −0.1 / −0.1 / −23)
MVDR bank        26.2    25.9    11.9
LCMV             26.2    25.8    11.9
Zero-forcing     26.7    26.6    11.9
```

*MVDR bank* = `w_mvdr()` called once per peak; each beam nulls the other two
implicitly because they're in `R`. *LCMV* makes those nulls explicit with
`C = [s₁ s₂ s₃]`, `f = e_k`. *Zero-forcing* uses only geometry — fine in white
noise, loses to the adaptive methods once noise is directional or reverberant.

## 6. LCMV

`src/pysdr_lcmv.py` — the "LCMV Beamformer" section.

![LCMV](plots/pysdr_lcmv.png)

- Ex 1: 8 elements, four interferers at −60°/−30°/0°/30°, two look directions
  at 15° and 60° (`f = [1, 1]`). Both constraints hold at exactly 1.0; the
  interferers land at −39 to −46 dB without being named — `R` does it.
- Ex 2: 18 elements, noise-only `R`, four unit-gain constraints across 15–30°
  and four zeros across 45–60°: a wide passband and stopband from one `w`,
  at the cost of 8 of 18 degrees of freedom.

MVDR is LCMV with one constraint; the three-source extractor is LCMV with
three.

## Takeaways for the microphone array

1. The DOA estimate is only half the job; the same `R⁻¹` that produced the
   scan gives the enhancement weights for free (`scan = 1/(sᴴR⁻¹s)` is the
   denominator of `w`).
2. Delay-and-sum against spatially white noise buys only `10·log10(M)` dB.
   The big wins come from nulling *directional* interference (MVDR/LCMV) and
   from a Wiener postfilter on top (`src/wiener.py` is exactly MVDR + Wiener).
3. With `M` mics you get `M−1` nulls; two mics → one null. Spend it on the
   loudest interferer.
4. Keep `d < λ/2` at the highest frequency you care about, or accept that
   some angle pairs become indistinguishable (the 2 kHz case above).
5. Acoustic signals are broadband: do everything per STFT bin with
   `s_k(f) = exp(2jπ f τ_k)`, then ISTFT.
