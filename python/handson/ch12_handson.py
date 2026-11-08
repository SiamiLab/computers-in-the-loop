r"""**Hands-on, Chapter 12.** These cells go with the Hands-on exercises at the end of
Chapter 12. Do each exercise on paper first and write down your prediction; then use the
matching section below to check it. Each section has sliders: change one thing at a time.
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11})
ORANGE, BLUE, TEAL, GRAY = "#E67E22", "#1D5FA8", "#1F6E78", "#6B7785"


def locus(A, B, Ks):
    """Roots of A(z) + K B(z) for each K (A, B as coefficient lists, highest power first)."""
    n = len(A) - 1
    Bp = np.r_[np.zeros(n + 1 - len(B)), B]
    return np.array([np.sort_complex(np.roots(np.asarray(A, float) + K * Bp)) for K in Ks])


def unit_circle(ax, lim=1.4):
    c = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(c), np.sin(c), color=GRAY, lw=1)
    ax.axhline(0, color=GRAY, lw=0.5)
    ax.axvline(0, color=GRAY, lw=0.5)
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)


# ---------------------------------------------------------------- H1
def h1_heater_locus(K=1.0):
    """H1: D(z)G(z) = 0.2K/((z-1)(z-0.6)), T = 0.5 s; locus, current poles, step response."""
    T = 0.5
    A, B = [1, -1.6, 0.6], [0.2]
    R = locus(A, B, np.linspace(0, 2.5, 600))
    p = np.roots(np.r_[1, -1.6, 0.6 + 0.2 * K])
    r = max(abs(p))
    tau = -T / np.log(r) if r < 1 else np.inf
    n = 40
    y = np.zeros(n + 1)
    for k in range(2, n + 1):           # closed loop 0.2K / (z^2 - 1.6 z + 0.6 + 0.2K), unit step
        y[k] = 1.6 * y[k - 1] - (0.6 + 0.2 * K) * y[k - 2] + 0.2 * K
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1, 1.6]})
    unit_circle(ax1)
    ax1.plot(R.real, R.imag, ".", color=BLUE, ms=1.5)
    ax1.plot(p.real, p.imag, "x", color=ORANGE, ms=12, mew=2.5)
    ax1.set_title(f"K = {K:.2f}: |z| = {r:.3f}", fontsize=11)
    ax2.plot(T * np.arange(n + 1), y, "o-", color=ORANGE, ms=3, lw=1)
    ax2.axhline(1, color=GRAY, lw=0.6)
    ax2.set_ylim(-0.2, 2.2)
    ax2.set_xlabel("t (s)")
    ax2.set_title(f"step response, envelope tau = {tau:.2f} s", fontsize=11)
    plt.show()


# ---------------------------------------------------------------- H2
def h2_gimbal_bode(K=4.0):
    """H2: Gw(w) = K (1 - w/20)(1 + w/241) / (w (1 + w/4.90)), T = 0.1 s; correct vs sign-flipped phase."""
    ww = np.logspace(-1, 3, 3000)
    s = 1j * ww
    Gw = K * (1 - s / 20) * (1 + s / 241) / (s * (1 + s / 4.90))
    ph = np.degrees(np.unwrap(np.angle(Gw)))
    ph_wrong = -90 - np.degrees(np.arctan(ww / 4.90)) + np.degrees(np.arctan(ww / 20)) \
        + np.degrees(np.arctan(ww / 241))
    ic = np.argmin(abs(abs(Gw) - 1))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True)
    ax1.semilogx(ww, 20 * np.log10(abs(Gw)), color=BLUE, lw=1.4)
    ax1.axhline(0, color=GRAY, lw=0.6)
    ax1.set_ylabel("|Gw| (dB)")
    ax1.set_title(f"K = {K:g}: crossover w = {ww[ic]:.2f} rad/s, "
                  f"PM {180 + ph[ic]:.1f} deg (sign-flipped reading {180 + ph_wrong[ic]:.1f} deg)",
                  fontsize=11)
    ax2.semilogx(ww, ph, color=BLUE, lw=1.4, label="correct phase")
    ax2.semilogx(ww, ph_wrong, "--", color=ORANGE, lw=1.4, label="zero 1 - w/20 read as a lead")
    ax2.axhline(-180, color=GRAY, lw=0.6)
    ax2.set_ylabel("phase (deg)")
    ax2.set_xlabel("w-plane frequency (rad/s)")
    ax2.legend(loc="lower left", fontsize=9)
    plt.show()


# ---------------------------------------------------------------- H3
def h3_reactor(K=1.7, T=1.0):
    """H3: Gp = a/(s - a), a = ln 2, ZOH, gain K: G(z) = (e^{aT} - 1)/(z - e^{aT})."""
    a = np.log(2)
    E = np.exp(a * T)
    pole = E - K * (E - 1)
    w = np.linspace(0, 2 * np.pi, 800)
    L = K * (E - 1) / (np.exp(1j * w) - E)
    ang = np.unwrap(np.angle(L + 1))
    ccw = int(np.round((ang[-1] - ang[0]) / (2 * np.pi)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    unit_circle(ax1, lim=2.6)
    Ks = np.linspace(0, 4, 200)
    ax1.plot(E - Ks * (E - 1), 0 * Ks, color=BLUE, lw=2)
    ax1.plot([E], [0], "x", color=GRAY, ms=10, mew=2)
    ax1.plot([pole], [0], "x", color=ORANGE, ms=12, mew=2.5)
    ax1.set_title(f"T = {T:.2f} s, K = {K:.2f}: pole {pole:.2f} "
                  f"({'stable' if abs(pole) < 1 else 'unstable'})", fontsize=11)
    ax2.plot(L.real, L.imag, color=BLUE, lw=1.4)
    ax2.plot([-1], [0], "+", color=ORANGE, ms=14, mew=2.5)
    ax2.set_aspect("equal")
    ax2.axhline(0, color=GRAY, lw=0.5)
    ax2.axvline(0, color=GRAY, lw=0.5)
    ax2.set_title(f"Nyquist of K G: encircles -1 {ccw:+d} times (counterclockwise +)", fontsize=11)
    plt.show()


# ---------------------------------------------------------------- H4
def h4_resonance(alpha=0.7):
    """H4: G = 0.1/((z-p)(z-conj p)), p = 0.8 + j0.5, D = K (z - alpha)/z; the locus near p."""
    p = 0.8 + 0.5j
    A = np.polymul([1, 0], [1, -1.6, 0.89])           # z (z^2 - 1.6 z + 0.89)
    B = [0.1, -0.1 * alpha]
    R = locus(A, B, np.linspace(0, 12, 1500))
    th_d = 180 + np.degrees(np.angle(p - alpha)) - np.degrees(np.angle(p)) - 90
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    unit_circle(ax, lim=1.3)
    c = np.linspace(0, 2 * np.pi, 300)
    ax.plot(abs(p) * np.cos(c), abs(p) * np.sin(c), ":", color=TEAL, lw=1)
    ax.plot(R.real, R.imag, ".", color=BLUE, ms=1.5)
    ax.plot([p.real, p.real, 0], [p.imag, -p.imag, 0], "x", color=GRAY, ms=10, mew=2)
    ax.plot([alpha], [0], "o", mfc="white", color=GRAY, ms=8)
    rad = np.radians(np.degrees(np.angle(p)))
    ax.annotate("", xy=(p.real + 0.25 * np.cos(rad), p.imag + 0.25 * np.sin(rad)),
                xytext=(p.real, p.imag), arrowprops=dict(arrowstyle="->", color=GRAY))
    ax.annotate("", xy=(p.real + 0.25 * np.cos(np.radians(th_d)), p.imag + 0.25 * np.sin(np.radians(th_d))),
                xytext=(p.real, p.imag), arrowprops=dict(arrowstyle="->", color=ORANGE, lw=2))
    diff = (th_d - np.degrees(np.angle(p))) % 360
    ax.set_title(f"alpha = {alpha:.2f}: departure {th_d:.1f} deg, {diff:.1f} deg from radial "
                 f"({'inward' if 90 < diff < 270 else 'outward'})", fontsize=10)
    plt.show()


try:
    from ipywidgets import interact, FloatSlider
    interact(h1_heater_locus, K=FloatSlider(value=1.0, min=0.02, max=2.4, step=0.02,
                                            description="K", continuous_update=False))
    interact(h2_gimbal_bode, K=FloatSlider(value=4.0, min=1.0, max=20.0, step=0.5,
                                           description="K", continuous_update=False))
    interact(h3_reactor, K=FloatSlider(value=1.7, min=0.2, max=4.0, step=0.05, description="K",
                                       continuous_update=False),
             T=FloatSlider(value=1.0, min=0.2, max=3.0, step=0.05, description="T (s)",
                           continuous_update=False))
    interact(h4_resonance, alpha=FloatSlider(value=0.7, min=-0.5, max=0.95, step=0.01,
                                             description="alpha", continuous_update=False))
except ImportError:
    h1_heater_locus()
    h2_gimbal_bode()
    h3_reactor()
    h4_resonance()
