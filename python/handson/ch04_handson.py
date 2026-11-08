r"""**Hands-on, Chapter 4.** These cells go with the Hands-on exercises at the end of
Chapter 4. Do each exercise on paper first and write down your prediction; then use the
matching section below to check it. Each section has sliders: change one thing at a time.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

plt.rcParams.update({"font.size": 11})
ORANGE, BLUE, TEAL, GRAY = "#E67E22", "#1D5FA8", "#1F6E78", "#6B7785"


# ---------------------------------------------------------------- H1
def h1_two_lakes(release=0.2):
    """H1: s(k+1) = (1-r) s(k) + 100 delta(k), l(k+1) = (1-r) l(k) + r s(k)."""
    n = 30
    s, l = np.zeros(n), np.zeros(n)
    for k in range(n - 1):
        s[k + 1] = (1 - release) * s[k] + (100.0 if k == 0 else 0.0)
        l[k + 1] = (1 - release) * l[k] + release * s[k]
    k = np.arange(n)
    kp = int(np.argmax(l))
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(k, s, "o-", color=BLUE, ms=4, lw=1, label="upper lake s(k)")
    ax.plot(k, l, "s-", color=ORANGE, ms=4, lw=1, label="lower lake l(k)")
    ax.set_xlabel("day k")
    ax.set_ylabel("excess volume")
    ax.set_title(f"release {release:.2f}/day: lower lake peaks on day {kp} at {l[kp]:.2f}; "
                 f"total outflow {release * l.sum():.1f} (of 100) in {n} days", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    plt.show()


# ---------------------------------------------------------------- H2
def h2_battery_pulse():
    """H2: pulse response of G(z) = 0.3/((z-0.5)(z-0.8)) against the classmate's 0.8^k - 0.5^k."""
    n = 16
    _, (g,) = signal.dimpulse(([0.3], [1, -1.3, 0.4], 1), n=n)
    g = g.ravel()
    k = np.arange(n)
    flawed = 0.8 ** k - 0.5 ** k
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(k, g, "o-", color=ORANGE, ms=6, lw=1, label="pulse response of G(z)")
    ax.plot(k, flawed, "s--", color=TEAL, ms=4, lw=1, label="classmate: 0.8^k - 0.5^k")
    ax.set_xlabel("k")
    ax.set_ylabel("temperature rise (C)")
    ax.set_title(f"hottest at k = {int(np.argmax(g))} (classmate: k = {int(np.argmax(flawed))})",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    plt.show()


# ---------------------------------------------------------------- H3
def h3_smoothing(a=0.5):
    """H3: y(k) = a y(k-1) + (1-a) x(k) on 0.5 m white noise, and on a 1 m climb."""
    rng = np.random.default_rng(0)
    n = 20000
    x = 0.5 * rng.standard_normal(n)
    y = signal.lfilter([1 - a], [1, -a], x)
    std = y[1000:].std()
    k = np.arange(120)
    climb = signal.lfilter([1 - a], [1, -a], np.ones(len(k)))
    inside = np.nonzero(np.abs(climb - 1) > 0.02)[0]
    ks = inside[-1] + 1 if len(inside) else 0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.8))
    ax1.plot(np.arange(300), x[:300], color=GRAY, lw=0.8, label="raw")
    ax1.plot(np.arange(300), y[:300], color=ORANGE, lw=1.4, label="smoothed")
    ax1.set_xlabel("k")
    ax1.set_ylabel("altitude noise (m)")
    ax1.set_title(f"a = {a:.3f}: output noise {std:.3f} m (input 0.5 m)", fontsize=11)
    ax1.legend(loc="upper right", fontsize=9)
    ax2.plot(k, climb, ".-", color=BLUE, ms=4, lw=1)
    ax2.axhspan(0.98, 1.02, color=TEAL, alpha=0.15)
    ax2.axvline(ks, color=GRAY, ls=":", lw=1)
    ax2.set_xlabel("k")
    ax2.set_title(f"1 m climb: within 2% from k = {ks}", fontsize=11)
    plt.show()


# ---------------------------------------------------------------- H4
def h4_shaping(m=0, c=0.0):
    """H4: G(z) = 1.44/((z-0.2)(z+0.8)); command 1/(1+c) for k < m, then 1 (m = 0: plain step)."""
    n = 16
    k = np.arange(n)
    u = np.where(k < m, 1 / (1 + c), 1.0)
    y = signal.lfilter([0, 0, 1.44], [1, 0.6, -0.16], u)
    ok_u = np.all((u >= -1e-12) & (u <= 1 + 1e-12))
    ok_y = np.all(y <= 1.02) and np.all(np.abs(y[6:] - 1) <= 0.02)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.step(k, u, where="post", color=GRAY, lw=1.4, label="command u(k)")
    ax.plot(k, y, "o-", color=ORANGE, ms=5, lw=1, label="head position y(k)")
    ax.axhspan(0.98, 1.02, color=BLUE, alpha=0.15, label="2% band")
    ax.axvline(6, color=GRAY, ls=":", lw=1)
    ax.set_ylim(-0.1, max(1.6, y.max() + 0.1, u.max() + 0.1))
    ax.set_xlabel("k")
    verdict = "PASS" if (ok_u and ok_y) else "FAIL"
    ax.set_title(f"m = {m}, c = {c:.3f}: max y {y.max():.3f}, max u {u.max():.3f}  ->  {verdict}",
                 fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    plt.show()


try:
    from ipywidgets import interact, FloatSlider, IntSlider, fixed
    interact(h1_two_lakes, release=FloatSlider(value=0.2, min=0.05, max=0.6, step=0.01,
                                               description="release", continuous_update=False))
    interact(h2_battery_pulse)
    interact(h3_smoothing, a=FloatSlider(value=0.5, min=0.0, max=0.99, step=0.002,
                                         readout_format=".3f", description="a",
                                         continuous_update=False))
    interact(h4_shaping, m=IntSlider(value=0, min=0, max=5, description="m",
                                     continuous_update=False),
             c=FloatSlider(value=0.0, min=-0.9, max=1.0, step=0.001, readout_format=".3f",
                           description="c", continuous_update=False))
except ImportError:
    h1_two_lakes()
    h2_battery_pulse()
    h3_smoothing()
    h4_shaping()
