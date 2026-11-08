r"""**A delay that is not a whole period.** A zero-order hold drives $1/(s+1)$, and the output
reaches the sampler $\Delta T$ seconds late, $0 < \Delta < 1$. The modified $z$-transform gives
$G(z,m)$ with $m = 1-\Delta$. Move $\Delta$ and watch the samples $c(kT)$ slide along the
undelayed curve $1 - e^{-t}$: they read it at $(k-\Delta)T$. On the right, the delay adds a pole
at $z = 0$ and a zero that moves with $m$; the DC gain stays 1.
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11})
ORANGE, BLUE, TEAL, GRAY = "#E67E22", "#1D5FA8", "#1F6E78", "#6B7785"


def explore(T=1.0, Delta=0.4):
    m = 1.0 - Delta
    p = np.exp(-T)
    b1 = 1.0 - np.exp(-m * T)                 # G(z,m) = (b1 z + b0) / (z (z - e^{-T}))
    b0 = np.exp(-m * T) - p

    # unit step through G(z,m): c(k) = p c(k-1) + b1 u(k-1) + b0 u(k-2), u = 1 for k >= 0
    t_end = max(5.0, 8 * T)
    n = int(np.floor(t_end / T + 1e-9)) + 1
    c = np.zeros(n)
    for k in range(1, n):
        c[k] = p * c[k - 1] + b1 + (b0 if k >= 2 else 0.0)
    k = np.arange(n)
    closed = np.where(k >= 1, 1 - np.exp(-(k - Delta) * T), 0.0)    # the curve read Delta*T late

    t = np.linspace(0, t_end, 800)
    c_delayed = np.where(t >= Delta * T, 1 - np.exp(-(t - Delta * T)), 0.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [2.1, 1]})
    ax1.plot(t, 1 - np.exp(-t), "--", color=GRAY, lw=1.4, label="no delay: $1 - e^{-t}$")
    ax1.plot(t, c_delayed, color=TEAL, lw=1.6, label=f"delayed output, $\\Delta T$ = {Delta * T:.2f} s")
    ax1.vlines(k * T, 0, c, color=ORANGE, lw=1)
    ax1.plot(k * T, c, "o", color=ORANGE, ms=6, label="samples $c(kT)$ from $G(z,m)$")
    ax1.set_xlim(0, t_end)
    ax1.set_ylim(-0.05, 1.15)
    ax1.set_xlabel("t (s)")
    ax1.set_ylabel("c")
    ax1.set_title(f"T = {T:g} s, delay $\\Delta$ = {Delta:.2f} T, so m = {m:.2f}", fontsize=11)
    ax1.legend(fontsize=10, loc="lower right")

    th = np.linspace(0, 2 * np.pi, 300)
    ax2.plot(np.cos(th), np.sin(th), color=GRAY, lw=1)
    ax2.plot([0, p], [0, 0], "x", color=BLUE, ms=11, mew=2.5, label="poles")
    zero = -b0 / b1
    if abs(zero) <= 1.25:
        ax2.plot([zero], [0], "o", mfc="white", mec=ORANGE, mew=2.5, ms=10, label="zero")
    else:
        ax2.annotate(f"zero at {zero:.2f}", xy=(-1.28, 0), xytext=(-1.2, 0.45), color=ORANGE,
                     fontsize=10, arrowprops=dict(arrowstyle="->", color=ORANGE))
    ax2.axhline(0, color="k", lw=0.5)
    ax2.axvline(0, color="k", lw=0.5)
    ax2.set_xlim(-1.3, 1.3)
    ax2.set_ylim(-1.3, 1.3)
    ax2.set_aspect("equal")
    ax2.set_xlabel("Re z")
    ax2.set_ylabel("Im z")
    ax2.set_title("poles and zero of G(z,m)", fontsize=11)
    ax2.legend(fontsize=10, loc="lower left")
    fig.tight_layout()

    dc = (b1 + b0) / (1 - p)
    print(f"G(z,m) = ({b1:.4f} z + {b0:.4f}) / (z (z - {p:.4f})),  m = 1 - Delta = {m:.2f}")
    print(f"zero at z = {zero:.3f};  DC gain G(1,m) = {dc:.4f}")
    print("c(kT), k = 0..4:", np.round(c[:5], 4),
          f"  (equal to 1 - e^-(k - Delta)T: largest difference {np.max(np.abs(c - closed)):.1e})")
    plt.show()


try:
    from ipywidgets import interact, FloatSlider
    interact(explore,
             T=FloatSlider(value=1.0, min=0.2, max=2.0, step=0.1, description="T (s)",
                           continuous_update=False),
             Delta=FloatSlider(value=0.4, min=0.05, max=0.95, step=0.05, description="Δ (of T)",
                               continuous_update=False))
except ImportError:
    explore()
