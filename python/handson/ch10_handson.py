r"""**Hands-on, Chapter 10.** These cells go with the Hands-on exercises at the end of
Chapter 10. Do each exercise on paper first and write down your prediction; then use the
matching section below to check it. Each section has sliders: change one thing at a time.
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11})
ORANGE, BLUE, TEAL, GRAY = "#E67E22", "#1D5FA8", "#1F6E78", "#6B7785"


def read_pole(z, T):
    """zeta, wn, tau of a z-plane pole, angle taken in the primary strip."""
    r, th = abs(z), abs(np.angle(z))
    L = np.hypot(np.log(r), th)
    return -np.log(r) / L, L / T, -T / np.log(r)


def proto_step(zeta, wn, t):
    """Unit step response of wn^2 / (s^2 + 2 zeta wn s + wn^2), 0 < zeta < 1."""
    wd = wn * np.sqrt(1 - zeta ** 2)
    return 1 - np.exp(-zeta * wn * t) * (np.cos(wd * t) + zeta * wn / wd * np.sin(wd * t))


# ---------------------------------------------------------------- H1
def h1_two_drones(T_A=0.005, T_B=0.05):
    """H1: loop A has poles 0.9 +- j0.1, loop B has 0.6 +- j0.45; time axis in seconds."""
    fig, ax = plt.subplots(figsize=(10, 3.8))
    t = np.linspace(0, 1.2, 2000)
    for z, T, col, name in [(0.9 + 0.1j, T_A, BLUE, "A"), (0.6 + 0.45j, T_B, ORANGE, "B")]:
        zeta, wn, tau = read_pole(z, T)
        Mp = np.exp(-np.pi * zeta / np.sqrt(1 - zeta ** 2))
        ax.plot(t, proto_step(zeta, wn, t), color=col, lw=1.4,
                label=f"{name}: T = {1000 * T:g} ms, zeta {zeta:.2f}, tau {tau:.3f} s, "
                      f"Ts {4 * tau:.2f} s, Mp {100 * Mp:.0f}%")
        k = np.arange(0, int(1.2 / T) + 1)
        ax.plot(k * T, proto_step(zeta, wn, k * T), "o", color=col, ms=3)
    ax.axhline(1, color=GRAY, lw=0.6)
    ax.set_xlabel("t (s)")
    ax.set_ylim(0, 1.4)
    ax.set_title("same z-plane poles, different sampling periods", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    plt.show()


# ---------------------------------------------------------------- H2
def h2_gimbal_mode(T=0.02):
    """H2: the mode r^k cos(k theta) of the pole -0.4 + j0.3, true angle vs misread angle."""
    z = -0.4 + 0.3j
    r = abs(z)
    th_true, th_wrong = np.angle(z), np.arctan(0.3 / -0.4)
    k = np.arange(0, 16)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.8), gridspec_kw={"width_ratios": [2, 1]})
    ax1.plot(k * T, r ** k * np.cos(k * th_true), "o-", color=ORANGE, ms=5, lw=1,
             label=f"true angle {th_true:.3f} rad")
    ax1.plot(k * T, r ** k * np.cos(k * abs(th_wrong)), "s--", color=TEAL, ms=4, lw=1,
             label=f"misread angle {abs(th_wrong):.3f} rad")
    ax1.axhline(0, color=GRAY, lw=0.6)
    ax1.set_xlabel("t (s)")
    zeta, wn, tau = read_pole(z, T)
    ax1.set_title(f"T = {1000 * T:g} ms: zeta {zeta:.2f}, wn {wn:.0f} rad/s, tau {1000 * tau:.0f} ms",
                  fontsize=11)
    ax1.legend(loc="upper right", fontsize=9)
    th = np.linspace(0, 2 * np.pi, 300)
    ax2.plot(np.cos(th), np.sin(th), color=GRAY, lw=1)
    ax2.plot([z.real, z.real], [z.imag, -z.imag], "x", color=ORANGE, ms=12, mew=2.5)
    ax2.plot([-z.real, -z.real], [z.imag, -z.imag], "x", color=TEAL, ms=9, mew=1.5)
    ax2.set_aspect("equal")
    ax2.set_xlim(-1.3, 1.3)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_title("true poles (orange), misread (teal)", fontsize=10)
    plt.show()


# ---------------------------------------------------------------- H3
def h3_tower_sway(T=0.3):
    """H3: the tower mode s = -0.5 +- j6 sampled every T; what the samples suggest."""
    s = -0.5 + 6j
    z = np.exp(s * T)
    zeta, wn, tau = read_pole(z, T)
    t = np.linspace(0, 12, 3000)
    k = np.arange(0, int(12 / T) + 1)
    th = np.angle(z)                              # primary-strip angle
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.8), gridspec_kw={"width_ratios": [2, 1]})
    ax1.plot(t, np.exp(-0.5 * t) * np.cos(6 * t), color=GRAY, lw=1, label="tower sway (true)")
    ax1.plot(t, np.exp(-0.5 * t) * np.cos(th / T * t), "--", color=BLUE, lw=1.4,
             label="mode the samples suggest")
    ax1.plot(k * T, np.exp(-0.5 * k * T) * np.cos(6 * k * T), "o", color=ORANGE, ms=5, label="samples")
    ax1.set_xlabel("t (s)")
    ax1.set_title(f"T = {T:.3f} s: angle {th:.3f} rad, apparent zeta {zeta:.2f}, "
                  f"wn {wn:.2f} rad/s (true 0.08, 6.02)", fontsize=11)
    ax1.legend(loc="upper right", fontsize=9)
    c = np.linspace(0, 2 * np.pi, 300)
    ax2.plot(np.cos(c), np.sin(c), color=GRAY, lw=1)
    ax2.plot([z.real, z.real], [z.imag, -z.imag], "x", color=ORANGE, ms=12, mew=2.5)
    ax2.set_aspect("equal")
    ax2.set_xlim(-1.3, 1.3)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_title("pole of the sampled mode", fontsize=10)
    plt.show()


# ---------------------------------------------------------------- H4
def h4_conveyor(K=0.55):
    """H4: belt speed G(z) = 1.5/(z - 0.25), T = 0.1 s, accumulator u_k = u_{k-1} + K e_k, ramp 2 m/s^2."""
    T, p, b, rate = 0.1, 0.25, 1.5, 2.0
    n = 40
    r = rate * T * np.arange(n + 1)
    y = np.zeros(n + 1)
    u = np.zeros(n + 1)
    for k in range(n):
        u[k] = (u[k - 1] if k >= 1 else 0.0) + K * (r[k] - y[k])
        y[k + 1] = p * y[k] + b * u[k]
    e = r - y
    poles = np.roots([1, -(1.25 - 1.5 * K), 0.25])
    if max(abs(poles)) >= 1:
        verdict, zeta = "unstable", float("nan")
    else:
        zp = poles[np.argmax(poles.imag)]
        zeta = read_pole(zp, T)[0]
        ok = zeta >= 0.5 and 0.1 / K <= 0.2
        verdict = "PASS" if ok else "FAIL"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.8), gridspec_kw={"width_ratios": [2, 1]})
    ax1.plot(T * np.arange(n + 1), e, "o-", color=ORANGE, ms=4, lw=1, label="following error (m/s)")
    ax1.axhline(0.2, color=TEAL, ls="--", lw=1, label="spec 0.2 m/s")
    ax1.axhline(0.1 / K, color=BLUE, lw=0.8, label=f"steady error {0.1 / K:.3f} m/s")
    ax1.set_ylim(-0.1, 0.6)
    ax1.set_xlabel("t (s)")
    ax1.set_title(f"K = {K:.3f}: zeta {zeta:.2f}, |z| {max(abs(poles)):.3f}: {verdict}", fontsize=11)
    ax1.legend(loc="upper right", fontsize=9)
    c = np.linspace(0, 2 * np.pi, 300)
    ax2.plot(np.cos(c), np.sin(c), color=GRAY, lw=1)
    ax2.plot(0.5 * np.cos(c), 0.5 * np.sin(c), ":", color=GRAY, lw=1)
    ax2.plot(poles.real, poles.imag, "x", color=ORANGE, ms=12, mew=2.5)
    ax2.set_aspect("equal")
    ax2.set_xlim(-1.3, 1.3)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_title("closed-loop poles", fontsize=10)
    plt.show()


try:
    from ipywidgets import interact, FloatSlider
    interact(h1_two_drones,
             T_A=FloatSlider(value=0.005, min=0.002, max=0.05, step=0.001, readout_format=".3f",
                             description="T_A (s)", continuous_update=False),
             T_B=FloatSlider(value=0.05, min=0.01, max=0.2, step=0.005, readout_format=".3f",
                             description="T_B (s)", continuous_update=False))
    interact(h2_gimbal_mode, T=FloatSlider(value=0.02, min=0.005, max=0.1, step=0.005,
                                           readout_format=".3f", description="T (s)",
                                           continuous_update=False))
    interact(h3_tower_sway, T=FloatSlider(value=0.3, min=0.05, max=1.2, step=0.001,
                                          readout_format=".3f", description="T (s)",
                                          continuous_update=False))
    interact(h4_conveyor, K=FloatSlider(value=0.55, min=0.05, max=1.7, step=0.01,
                                        readout_format=".2f", description="K",
                                        continuous_update=False))
except ImportError:
    h1_two_drones()
    h2_gimbal_mode()
    h3_tower_sway()
    h4_conveyor()
