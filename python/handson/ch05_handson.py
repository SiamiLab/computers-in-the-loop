r"""**Hands-on, Chapter 5.** These cells go with the Hands-on exercises at the end of
Chapter 5. Do each exercise on paper first and write down your prediction; then use the
matching section below to check it. Each section has sliders: change one thing at a time.
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11})
ORANGE, BLUE, TEAL, GRAY = "#E67E22", "#1D5FA8", "#1F6E78", "#6B7785"


def run_state_space(A, B, C, D, e):
    """x(k+1) = A x(k) + B e(k), m(k) = C x(k) + D e(k), from x(0) = 0; returns m and the states."""
    x = np.zeros(A.shape[0])
    m, xs = [], []
    for ek in e:
        xs.append(x.copy())
        m.append(C @ x + D * ek)
        x = A @ x + B * ek
    return np.array(m), np.array(xs)


# ---------------------------------------------------------------- H1
def h1_ring(T=0.05):
    """H1: e(t) = exp(-2t) cos(4 pi t) sampled every T; poles of E(z) at exp(-2T) exp(+-j 4 pi T)."""
    t = np.linspace(0, 3, 3000)
    k = np.arange(int(3 / T) + 1)
    r, th = np.exp(-2 * T), 4 * np.pi * T
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.8), gridspec_kw={"width_ratios": [2, 1]})
    ax1.plot(t, np.exp(-2 * t) * np.cos(4 * np.pi * t), color=GRAY, lw=1.2, label="e(t)")
    ax1.plot(k * T, np.exp(-2 * k * T) * np.cos(4 * np.pi * k * T), "o-", color=ORANGE, ms=5,
             lw=1, label="samples")
    ax1.axhline(0, color="k", lw=0.5)
    ax1.set_xlabel("t (s)")
    ax1.set_title(f"T = {T:g} s: angle {np.degrees(th):.1f} deg per sample "
                  f"({360 / np.degrees(th):.2f} samples per cycle), radius {r:.3f}", fontsize=10)
    ax1.legend(loc="upper right", fontsize=9)
    circ = np.linspace(0, 2 * np.pi, 300)
    ax2.plot(np.cos(circ), np.sin(circ), color=GRAY, lw=1)
    ax2.plot(r * np.cos(th) * np.ones(2), r * np.sin(th) * np.array([1, -1]), "x", color=ORANGE,
             ms=12, mew=2.5)
    ax2.set_aspect("equal")
    ax2.set_xlim(-1.3, 1.3)
    ax2.set_ylim(-1.3, 1.3)
    ax2.set_title("poles of E(z)", fontsize=11)
    plt.show()


# ---------------------------------------------------------------- H2
def h2_programs():
    """H2: pulse response of D(z) = (2z^2 - 1.5z + 0.12)/(z^2 - z + 0.21): two programs."""
    A = np.array([[0, 1], [-0.21, 1.0]])
    B = np.array([0, 1.0])
    e = np.zeros(15)
    e[0] = 1
    m_good, _ = run_state_space(A, B, np.array([-0.3, 0.5]), 2.0, e)
    m_bad, _ = run_state_space(A, B, np.array([0.12, -1.5]), 0.0, e)
    # the difference equation itself: m(k) = m(k-1) - 0.21 m(k-2) + 2 e(k) - 1.5 e(k-1) + 0.12 e(k-2)
    m_de = np.zeros(15)
    for k in range(15):
        m_de[k] = ((m_de[k - 1] if k >= 1 else 0) - 0.21 * (m_de[k - 2] if k >= 2 else 0)
                   + 2 * e[k] - 1.5 * (e[k - 1] if k >= 1 else 0) + 0.12 * (e[k - 2] if k >= 2 else 0))
    k = np.arange(15)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(k, m_de, "o", color=ORANGE, ms=9, mfc="none", mew=2, label="difference equation of D(z)")
    ax.plot(k, m_good, ".", color=BLUE, ms=8, label="corrected program")
    ax.plot(k, m_bad, "s--", color=TEAL, ms=4, lw=1, label="classmate's program")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("k")
    ax.set_title("pulse response: first samples " + ", ".join(f"{v:.3f}" for v in m_de[:3]),
                 fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    plt.show()


# ---------------------------------------------------------------- H3
def h3_hidden_pole(c=1.0):
    """H3: G(z) = (z - c)/((z - 0.5)(z - 1.25)) in control canonical form, unit pulse input."""
    A = np.array([[0, 1], [-0.625, 1.75]])
    B = np.array([0, 1.0])
    n = 60
    e = np.zeros(n)
    e[0] = 1
    m, xs = run_state_space(A, B, np.array([-c, 1.0]), 0.0, e)
    k = np.arange(n)
    over = np.nonzero(np.abs(xs[:, 1]) > 32767)[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.8))
    ax1.plot(k, m, "o-", color=ORANGE, ms=4, lw=1)
    ax1.set_ylim(-3, 3)
    ax1.axhline(0, color="k", lw=0.5)
    ax1.set_xlabel("k")
    ax1.set_title(f"c = {c:.3f}: output m(k)", fontsize=11)
    ax2.semilogy(k, np.abs(xs[:, 0]) + 1e-12, color=BLUE, lw=1.4, label="|x1|")
    ax2.semilogy(k, np.abs(xs[:, 1]) + 1e-12, "--", color=TEAL, lw=1.4, label="|x2|")
    ax2.axhline(32767, color=GRAY, ls=":", lw=1, label="16-bit limit")
    ax2.set_xlabel("k")
    ax2.set_ylim(1e-3, 1e6)
    ax2.set_title("stored numbers" + (f": x2 overflows at k = {over[0]}" if len(over) else ""),
                  fontsize=11)
    ax2.legend(loc="lower right", fontsize=9)
    plt.show()


# ---------------------------------------------------------------- H4
def h4_hold(T_ms=0.3, held="start of period"):
    """H4: x(t) = 10 sin(2 pi t) mm held every T; gap |x - xbar| over one cycle."""
    T = T_ms * 1e-3
    n = int(np.ceil(1 / T))
    t = np.linspace(0, 1, 200001)
    k = np.floor(t / T)
    tk = (k + 0.5) * T if held.startswith("middle") else k * T
    gap = np.abs(10 * np.sin(2 * np.pi * t) - 10 * np.sin(2 * np.pi * tk))
    worst = gap.max()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.8))
    win = t <= 12 * T
    ax1.plot(t[win] * 1e3, 10 * np.sin(2 * np.pi * t[win]), color=GRAY, lw=1.2, label="path x(t)")
    ax1.plot(t[win] * 1e3, 10 * np.sin(2 * np.pi * tk[win]), color=ORANGE, lw=1.6, label="held value")
    ax1.set_xlabel("t (ms)")
    ax1.set_ylabel("mm")
    ax1.set_title(f"first {12} periods ({n} updates per cycle)", fontsize=11)
    ax1.legend(loc="upper left", fontsize=9)
    ax2.plot(t, gap * 1e3, color=BLUE, lw=0.6)
    ax2.axhline(20, color=TEAL, ls="--", lw=1, label="spec 0.02 mm")
    ax2.set_xlabel("t (s)")
    ax2.set_ylabel("gap (micrometers)")
    ok = (worst <= 0.02) and (T_ms >= 0.5)
    ax2.set_title(f"{held}, T = {T_ms:.3f} ms: worst gap {worst * 1e3:.2f} um -> "
                  f"{'PASS' if ok else 'FAIL'}", fontsize=10)
    ax2.legend(loc="upper right", fontsize=9)
    plt.show()


try:
    from ipywidgets import interact, FloatSlider, Dropdown
    interact(h1_ring, T=FloatSlider(value=0.05, min=0.01, max=0.6, step=0.01, description="T (s)",
                                    continuous_update=False))
    interact(h2_programs)
    interact(h3_hidden_pole, c=FloatSlider(value=1.0, min=0.0, max=1.5, step=0.01,
                                           description="c", continuous_update=False))
    interact(h4_hold, T_ms=FloatSlider(value=0.3, min=0.1, max=1.0, step=0.01, readout_format=".2f",
                                       description="T (ms)", continuous_update=False),
             held=Dropdown(options=["start of period", "middle of period"],
                           value="start of period", description="hold"))
except ImportError:
    h1_ring()
    h2_programs()
    h3_hidden_pole()
    h4_hold()
