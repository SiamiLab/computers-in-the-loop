r"""**Holds on a sine.** The samples of $e(t) = \sin t$ (so $\omega = 1$ rad/s and $\omega T = T$)
are rebuilt by the zero-order hold and by the fractional-order hold
$\bar e(t) = e(kT) + \beta\,[e(kT) - e((k-1)T)](t - kT)/T$: $\beta = 0$ is the ZOH,
$\beta = 1$ the FOH. At $T = 0.5$ s the ZOH lags by $14.3^\circ$ and the FOH by only $2.1^\circ$.
Grow $T$ and watch the staircase fall behind, while the FOH overshoots where the sine bends;
move $\beta$ to trade lag against overshoot.
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11})
ORANGE, BLUE, TEAL, GRAY = "#E67E22", "#1D5FA8", "#1F6E78", "#6B7785"
W0 = 1.0          # rad/s
T_END = 12.0      # seconds shown


def _G_hold(w, T, beta):
    """G_hb(jw) = (1 - b e^{-jwT})(1 - e^{-jwT})/(jw) + b/(T (jw)^2) (1 - e^{-jwT})^2."""
    s = 1j * w
    x = np.exp(-T * s)
    return (1 - beta * x) * (1 - x) / s + beta / (T * s ** 2) * (1 - x) ** 2


def _gain_phase(T, beta):
    """Gain over T and phase in degrees at w = W0, phase unwrapped from w = 0."""
    w = np.linspace(1e-4, W0, 400)
    G = _G_hold(w, T, beta)
    return abs(G[-1]) / T, np.degrees(np.unwrap(np.angle(G))[-1]) + 0.0


def _hold(t, T, beta):
    k = np.floor(t / T + 1e-12)
    ek = np.sin(W0 * k * T)
    ekm1 = np.where(k >= 1, np.sin(W0 * (k - 1) * T), 0.0)   # e = 0 before t = 0
    return ek + beta * (ek - ekm1) * (t - k * T) / T


def explore(T=0.5, beta=1.0):
    T, beta = round(T, 10), round(beta, 10)
    t = np.linspace(0, T_END, 3001)
    e = np.sin(W0 * t)
    zoh = _hold(t, T, 0.0)
    frac = _hold(t, T, beta)
    tk = T * np.arange(int(T_END / T) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.0),
                                   gridspec_kw={"width_ratios": [1.8, 1]})
    ax1.plot(t, e, color=GRAY, lw=1.4, label=r"$e(t) = \sin t$")
    ax1.plot(t, zoh, color=BLUE, lw=1.4, label="ZOH")
    if beta > 0:
        ax1.plot(t, frac, color=ORANGE, lw=1.8,
                 label="FOH" if beta == 1 else rf"fractional, $\beta$ = {beta:g}")
    ax1.plot(tk, np.sin(W0 * tk), "o", color="k", ms=3.5, label="samples")
    ax1.set_xlim(0, T_END)
    lo = min(-1.3, frac.min() - 0.1)
    hi = max(1.3, frac.max())
    ax1.set_ylim(lo, hi + 0.25 * (hi - lo) + 0.2)          # room for the legend
    ax1.set_xlabel("t (s)")
    ax1.set_ylabel(r"$\bar e(t)$")
    ax1.set_title(rf"T = {T:g} s ($\omega T$ = {W0 * T:g}), $\beta$ = {beta:g}", fontsize=11)
    ax1.legend(fontsize=10, loc="upper right", ncol=4)

    ws = 2 * np.pi / T
    w = np.linspace(1e-4, 2 * ws, 1500)
    ax2.plot(w / ws, np.abs(_G_hold(w, T, 0.0)) / T, color=BLUE, lw=1.6, label="ZOH")
    if beta > 0:
        ax2.plot(w / ws, np.abs(_G_hold(w, T, beta)) / T, color=ORANGE, lw=1.8,
                 label=rf"$\beta$ = {beta:g}")
    ax2.axvline(W0 / ws, color=TEAL, ls="--", lw=1.4, label=r"$\omega = 1$ rad/s")
    ax2.set_xlim(0, 2)
    ax2.set_ylim(0, 1.75)
    ax2.set_xlabel(r"$\omega / \omega_s$")
    ax2.set_ylabel(r"$|G_h(j\omega)| / T$")
    ax2.set_title("hold gain", fontsize=11)
    ax2.legend(fontsize=10, loc="upper right")
    fig.tight_layout()

    g0, ph0 = _gain_phase(T, 0.0)
    gb, phb = _gain_phase(T, beta)
    print(f"omega T = {W0 * T:g}  (omega = {W0 * T / (2 * np.pi):.3f} omega_s)")
    print(f"ZOH:          gain {g0:.3f} T, phase {round(ph0, 1) + 0.0:6.1f} deg")
    print(f"beta = {beta:<4g}   gain {gb:.3f} T, phase {round(phb, 1) + 0.0:6.1f} deg;  "
          f"low-frequency delay (1 - beta) T/2 = {(1 - beta) * T / 2:.3f} s")
    after = t >= 2 * T
    print(f"largest |e(t) - hold output| after t = 2T: ZOH {np.max(np.abs(e - zoh)[after]):.3f}, "
          f"beta = {beta:g}: {np.max(np.abs(e - frac)[after]):.3f}")
    plt.show()


try:
    from ipywidgets import interact, FloatSlider
    interact(explore,
             T=FloatSlider(value=0.5, min=0.1, max=2.5, step=0.05, description="T (s)",
                           continuous_update=False),
             beta=FloatSlider(value=1.0, min=0.0, max=1.0, step=0.1, description="beta",
                              continuous_update=False))
except ImportError:
    explore()
