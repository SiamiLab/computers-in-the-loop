#!/usr/bin/env python3
"""Chapter 15 companion: implementing digital controllers.

Run it with
    python3 ch15_implementation.py
It prints the main numbers of the chapter and saves six figures (PNG) in a folder
ch15_figures/ next to this file:

  1. ch15_pm_vs_T.png        phase margin against T, with and without the anti-aliasing filter
  2. ch15_delay_steps.png    step responses with and without one sample of computation delay
  3. ch15_noise_gain.png     round-off noise in a first-order filter: formula against simulation
  4. ch15_coef_rounding.png  poles of a 4th-order filter after rounding: direct form vs cascade
  5. ch15_limit_cycles.png   deadband and limit cycle of a rounded first-order recursion
  6. ch15_windup.png         PI control with a saturating actuator: windup and anti-windup

Requires numpy, scipy and matplotlib. Change the parameters in the CAPITAL constants at the top
of each section to experiment.
"""
import cmath
import math
import os

import numpy as np
from scipy import linalg, optimize

import matplotlib
matplotlib.use("Agg")            # draw to files, no window needed
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "ch15_figures")
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------------------------
# Tools: zero-order-hold discretization and the phase margin of a sampled loop
# ---------------------------------------------------------------------------------------------
def zoh(A, B, T):
    """Exact ZOH discretization: Phi = e^{AT}, Gamma = int_0^T e^{As} ds B (one matrix exponential)."""
    n = A.shape[0]
    M = np.zeros((n + 1, n + 1))
    M[:n, :n] = A
    M[:n, n:] = B
    E = linalg.expm(M * T)
    return E[:n, :n], E[:n, n:]


def plant_with_prefilter(wf=None):
    """State space of G_p(s) = 1/(s(s+1)), optionally followed by the Butterworth prefilter
    H_f(s) = wf^2 / (s^2 + sqrt(2) wf s + wf^2) on the measurement."""
    if wf is None:
        A = np.array([[0.0, 1.0], [0.0, -1.0]])
        B = np.array([[0.0], [1.0]])
        C = np.array([[1.0, 0.0]])
    else:
        A = np.array([[0, 1, 0, 0],
                      [0, -1, 0, 0],
                      [0, 0, 0, 1],
                      [wf ** 2, 0, -wf ** 2, -math.sqrt(2) * wf]], dtype=float)
        B = np.array([[0.0], [1.0], [0.0], [0.0]])
        C = np.array([[0.0, 0.0, 1.0, 0.0]])
    return A, B, C


def loop_gain(T, w, K=1.0, wf=None, delay=0):
    """K z^-delay times the pulse transfer function of ZOH, plant (and prefilter), at z = e^{jwT}."""
    A, B, C = plant_with_prefilter(wf)
    Phi, Gam = zoh(A, B, T)
    z = cmath.exp(1j * w * T)
    G = (C @ np.linalg.solve(z * np.eye(A.shape[0]) - Phi, Gam))[0, 0]
    return K * G / z ** delay


def phase_margin(T, K=1.0, wf=None, delay=0):
    """Find the crossover |L| = 1 below the Nyquist frequency and return (w_c, PM in degrees)."""
    wc = optimize.brentq(lambda w: abs(loop_gain(T, w, K, wf, delay)) - 1, 1e-3, math.pi / T * 0.999)
    pm = 180 + math.degrees(cmath.phase(loop_gain(T, wc, K, wf, delay)))
    if pm > 180:
        pm -= 360
    return wc, pm


# ---------------------------------------------------------------------------------------------
# 1. Sampling period and anti-aliasing prefilter
# ---------------------------------------------------------------------------------------------
ATTENUATION_AT_NYQUIST = 0.1     # prefilter gain at w_s/2 (0.1 = -20 dB)


def section_sampling_period():
    print("\n1. Sampling period and prefilter (plant 1/(s(s+1)), K = 1)")
    wc = math.sqrt((math.sqrt(5) - 1) / 2)          # analog crossover
    pm_analog = 90 - math.degrees(math.atan(wc))
    print(f"  analog design: w_c = {wc:.3f} rad/s, PM = {pm_analog:.1f} deg")
    # prefilter cutoff from |H_f(j w_N)| = 1/sqrt(1 + (w_N/w_f)^4) = ATTENUATION
    ratio = (1 / ATTENUATION_AT_NYQUIST ** 2 - 1) ** 0.25
    Ts = np.arange(0.02, 0.401, 0.02)
    pm_hold, pm_both = [], []
    for T in Ts:
        wf = (math.pi / T) / ratio
        pm_hold.append(phase_margin(T)[1])
        pm_both.append(phase_margin(T, wf=wf)[1])
    for T in (0.05, 0.1, 0.2):
        wf = (math.pi / T) / ratio
        print(f"  T = {T:4.2f} s: w_f = {wf:6.2f} rad/s, PM hold only = {phase_margin(T)[1]:.1f} deg,"
              f" hold + prefilter = {phase_margin(T, wf=wf)[1]:.1f} deg")
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(Ts, pm_hold, "o-", label="hold only (exact)")
    ax.plot(Ts, pm_both, "s-", label="hold and prefilter (exact)")
    ax.plot(Ts, pm_analog - np.degrees(wc * Ts / 2), "--", label="estimate: hold = delay T/2")
    k_pf = math.sqrt(2) * ratio / math.pi            # prefilter acts like a delay of k_pf * T
    ax.plot(Ts, pm_analog - np.degrees((0.5 + k_pf) * wc * Ts), "--",
            label=f"estimate: hold + prefilter = delay {0.5 + k_pf:.2f} T")
    ax.axhline(pm_analog, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("T (s)")
    ax.set_ylabel("phase margin (deg)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, "ch15_pm_vs_T.png")


# ---------------------------------------------------------------------------------------------
# 2. Computation delay in the running example (ZOH + 1/(s(s+1)), T = 1 s, D(z) = K)
# ---------------------------------------------------------------------------------------------
def step_response(K, delay, n=40):
    """c(k) of the unity-feedback loop with D(z) = K z^-delay, unit step reference."""
    a = math.exp(-1)
    b1, b0 = a, 1 - 2 * a                 # G(z) = (b1 z + b0) / ((z - 1)(z - a))
    c = np.zeros(n)
    m = np.zeros(n)                       # control applied to the plant
    for k in range(n):
        c[k] = ((1 + a) * c[k - 1] if k >= 1 else 0) - (a * c[k - 2] if k >= 2 else 0) \
            + (b1 * m[k - 1] if k >= 1 else 0) + (b0 * m[k - 2] if k >= 2 else 0)
        if delay:                         # m(k) computed at k is written at k + 1
            m[k] = K * (1 - c[k - 1]) if k >= 1 else 0.0
        else:
            m[k] = K * (1 - c[k])
    return c


def largest_stable_gain(delay_fraction):
    """Largest K for D(z) = K when the output leaves delay_fraction * T after the sample (T = 1).
    State space with the delayed input: x(k+1) = Phi x + G0 m(k) + G1 m(k-1)."""
    A, B, C = plant_with_prefilter(None)
    T, tau = 1.0, delay_fraction
    Phi, _ = zoh(A, B, T)
    P1, G0 = zoh(A, B, T - tau)
    _, Gt = zoh(A, B, tau)
    G1 = P1 @ Gt
    lo, hi = 0.0, 5.0
    for _ in range(60):                   # bisection on the spectral radius
        K = (lo + hi) / 2
        Acl = np.zeros((3, 3))
        Acl[:2, :2] = Phi - K * G0 @ C
        Acl[:2, 2:] = G1
        Acl[2, :2] = -K * C
        if max(abs(np.linalg.eigvals(Acl))) < 1:
            lo = K
        else:
            hi = K
    return lo


def section_delay():
    print("\n2. Computation delay (running example, T = 1 s)")
    for frac in (1e-9, 0.1, 0.2, 0.5, 1 - 1e-9):
        print(f"  output {frac:4.2f} T after the sample: stable for 0 < K < {largest_stable_gain(frac):.2f}")
    for K, d in [(0.5, 0), (0.5, 1), (0.3, 1)]:
        wc, pm = phase_margin(1.0, K=K, delay=d)
        c = step_response(K, d, 80)
        print(f"  K = {K}, delay = {d}: PM = {pm:.1f} deg, overshoot = {100 * (c.max() - 1):.1f} %")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for K, d, mk in [(0.5, 0, "o"), (0.5, 1, "s"), (0.3, 1, "^")]:
        ax.plot(step_response(K, d), mk + "-", ms=3, label=f"K = {K}, " + ("one sample of delay" if d else "no delay"))
    ax.axhline(1, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("k")
    ax.set_ylabel("c(kT)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, "ch15_delay_steps.png")


# ---------------------------------------------------------------------------------------------
# 3. Quantization noise through a first-order filter m(k) = Q[a m(k-1) + (1-a) e(k)]
# ---------------------------------------------------------------------------------------------
def Q(x):
    """Round to the nearest integer (units of q), ties away from zero."""
    return math.floor(abs(x) + 0.5) * (1 if x >= 0 else -1)


def section_noise():
    print("\n3. Round-off noise in m(k) = Q[a m(k-1) + (1-a) e(k)]   (units of q)")
    rng = np.random.default_rng(1)
    poles = [0.5, 0.8, 0.9, 0.92, 0.95, 0.98]
    formula, simulated = [], []
    for a in poles:
        e = rng.uniform(-500, 500, 60000)          # a busy input, so the error looks random
        mq, mi = 0.0, 0.0
        err = []
        for ek in e:
            mq = Q(a * mq + (1 - a) * ek)          # what the processor computes
            mi = a * mi + (1 - a) * ek             # infinite precision
            err.append(mq - mi)
        formula.append(1 / (1 - a ** 2) / 12)      # noise gain times q^2/12
        simulated.append(np.var(err[2000:]))
        print(f"  a = {a:4.2f}: variance formula {formula[-1]:7.3f}, simulated {simulated[-1]:7.3f}")
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.semilogy(poles, formula, "o-", label=r"$\frac{q^2}{12}\cdot\frac{1}{1-a^2}$")
    ax.semilogy(poles, simulated, "s", label="simulation with rounding")
    ax.set_xlabel("pole a")
    ax.set_ylabel(r"output noise variance (units of $q^2$)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    save(fig, "ch15_noise_gain.png")


# ---------------------------------------------------------------------------------------------
# 4. Coefficient rounding: 4th-order filter with poles e^{-kT}, k = 1..4
# ---------------------------------------------------------------------------------------------
T_COEF = 0.05          # sampling period (s); try 0.1 and 0.02
DECIMALS = 3           # decimals kept in each stored coefficient


def section_coefficients():
    print(f"\n4. Coefficient rounding, poles e^(-kT), T = {T_COEF} s, {DECIMALS} decimals")
    p = np.exp(-T_COEF * np.arange(1, 5))
    a = np.poly(p)                                   # direct form: one 4th-order polynomial
    a_rounded = np.round(a, DECIMALS)
    direct = np.roots(a_rounded)
    cascade = np.round(p, DECIMALS)                  # first-order sections: coefficient = pole
    print(f"  exact poles         {np.round(p, 4)}")
    print(f"  A(1) exact {np.polyval(a, 1):.3e}, after rounding {np.polyval(a_rounded, 1):.3e}")
    print(f"  direct form poles   {np.round(direct, 4)}   max |z| = {max(abs(direct)):.4f}")
    print(f"  cascade poles       {cascade}   largest move = {max(abs(cascade - p)):.4f}")
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    th = np.linspace(-0.4, 0.4, 200)
    ax.plot(np.cos(th), np.sin(th), color="gray", lw=1, label="|z| = 1")
    ax.plot(p, 0 * p, "x", ms=9, mew=2, label="exact")
    ax.plot(direct.real, direct.imag, "x", ms=9, mew=2, label="direct form, rounded")
    ax.plot(cascade, 0 * cascade, "+", ms=11, mew=1.5, label="cascade, rounded")
    ax.set_aspect("equal")
    ax.set_xlim(0.6, 1.1)
    ax.set_ylim(-0.2, 0.2)
    ax.legend(fontsize=8, loc="center left")
    ax.grid(alpha=0.3)
    save(fig, "ch15_coef_rounding.png")


# ---------------------------------------------------------------------------------------------
# 5. Deadband and limit cycles: x(k+1) = Q[a x(k)]
# ---------------------------------------------------------------------------------------------
def section_limit_cycles():
    print("\n5. Rounded first-order recursion x(k+1) = Q[a x(k)], x(0) = 10")
    n = 15
    trunc = lambda x: math.floor(abs(x)) * (1 if x >= 0 else -1)   # magnitude truncation
    runs = {}
    for label, a, quant in [("a = 0.92, rounding", 0.92, Q), ("a = -0.92, rounding", -0.92, Q),
                            ("a = 0.92, truncation toward 0", 0.92, trunc)]:
        x = [10]
        for _ in range(n - 1):
            x.append(quant(a * x[-1]))
        runs[label] = x
        print(f"  {label:30s}: {x}")
    print(f"  deadband for |a| = 0.92: |x| <= 1/(2(1 - 0.92)) = {1 / (2 * 0.08):.2f}")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    k = np.arange(n)
    ax.plot(k, 10 * 0.92 ** k, color="gray", label="no rounding")
    for (label, x), dx, mk in zip(runs.items(), (0, 0.2, -0.2), ("o", "s", "^")):
        ax.plot(k + dx, x, mk, label=label)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("k")
    ax.set_ylabel("x(k) / q")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, "ch15_limit_cycles.png")


# ---------------------------------------------------------------------------------------------
# 6. Saturation and windup: plant 1/(s+1), digital PI, |v| <= M_MAX
# ---------------------------------------------------------------------------------------------
T_PI, KP, KI, M_MAX, T_TRACK = 0.1, 4.0, 4.0, 1.2, 1.0


def pi_loop(mode, t_end=8.0):
    """mode: 'none' (no limit), 'windup' (plain PI), 'back' (back-calculation), 'clamp'."""
    p = math.exp(-T_PI)                  # exact ZOH model of 1/(s+1)
    c, I = 0.0, 0.0
    C, Iout, V = [], [], []
    for _ in range(int(round(t_end / T_PI)) + 1):
        e = 1.0 - c                      # unit step reference
        m = KP * e + I                   # PI output, before the limit
        v = m if mode == "none" else max(-M_MAX, min(M_MAX, m))
        C.append(c)
        Iout.append(I)
        V.append(v)
        if mode == "back":               # back-calculation anti-windup
            I += KI * T_PI * e + (T_PI / T_TRACK) * (v - m)
        elif mode == "clamp":            # conditional integration
            if abs(m) <= M_MAX or m * e < 0:
                I += KI * T_PI * e
        else:
            I += KI * T_PI * e
        c = p * c + (1 - p) * v          # plant update over one period
    return np.array(C), np.array(Iout), np.array(V)


def section_windup():
    print(f"\n6. PI with |v| <= {M_MAX}: plant 1/(s+1), T = {T_PI} s, Kp = Ki = {KP}")
    names = {"none": "no limit", "windup": "limit, plain PI", "back": "limit, back-calculation",
             "clamp": "limit, conditional integration"}
    colors = {"none": "gray", "windup": "tab:red", "back": "tab:blue", "clamp": "tab:green"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for mode in ("none", "windup", "back", "clamp"):
        C, I, V = pi_loop(mode, t_end=12.0)
        t = T_PI * np.arange(len(C))
        settle = T_PI * (max(k for k in range(len(C)) if abs(C[k] - 1) > 0.02) + 1)
        print(f"  {names[mode]:32s}: peak c = {C.max():.3f}, largest I = {I.max():.2f},"
              f" 2% settling = {settle:.1f} s")
        keep = t <= 8.0
        ax1.plot(t[keep], C[keep], color=colors[mode], label=names[mode])
        if mode != "none":
            ax2.plot(t[keep], I[keep], color=colors[mode], label=names[mode])
    ax1.axhline(1, color="gray", lw=0.8, ls=":")
    ax2.axhline(M_MAX, color="gray", lw=0.8, ls="--")
    ax1.set_title("output c(kT)")
    ax2.set_title("integrator state I(k)")
    for ax in (ax1, ax2):
        ax.set_xlabel("t (s)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    save(fig, "ch15_windup.png")


if __name__ == "__main__":
    section_sampling_period()
    section_delay()
    section_noise()
    section_coefficients()
    section_limit_cycles()
    section_windup()
    print(f"\nFigures are in {OUT}")
