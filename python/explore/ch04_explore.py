r"""**Pole positions and the inverse sequence.** Take $E(z) = z/\big((z-p_1)(z-p_2)\big)$.
Partial fractions of $E(z)/z$ give $e_k = k_1 p_1^k + k_2 p_2^k$ with
$k_1 = 1/(p_1 - p_2)$, $k_2 = 1/(p_2 - p_1)$; the defaults $p_1 = 1$, $p_2 = 2$ give
$e_k = 2^k - 1$. Move the poles: each one contributes its own mode $p_i^k$, a negative pole
flips sign every sample, and one pole outside the unit circle is enough to make $e_k$ grow.
Set $p_1 = p_2$ to see the repeated-pole case, $e_k = k\,p^{k-1}$.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

plt.rcParams.update({"font.size": 11})
ORANGE, BLUE, TEAL, GRAY = "#E67E22", "#1D5FA8", "#1F6E78", "#6B7785"
NK = 13       # k = 0..12


def explore(p1=1.0, p2=2.0):
    p1, p2 = round(p1, 10), round(p2, 10)
    k = np.arange(NK)
    # long division: the impulse response of z^-1 / (1 - (p1 + p2) z^-1 + p1 p2 z^-2)
    den = [1.0, -(p1 + p2), p1 * p2]
    e_div = signal.lfilter([0.0, 1.0, 0.0], den, np.r_[1.0, np.zeros(NK - 1)])
    repeated = abs(p1 - p2) < 1e-9
    if repeated:
        with np.errstate(divide="ignore", invalid="ignore"):
            e_pf = np.where(k == 0, 0.0, k * p1 ** np.maximum(k - 1, 0))
        modes = None
    else:
        k1, k2 = 1 / (p1 - p2), 1 / (p2 - p1)
        m1, m2 = k1 * p1 ** k, k2 * p2 ** k
        e_pf = m1 + m2
        modes = (m1, m2)

    rho = max(abs(p1), abs(p2))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.0),
                                   gridspec_kw={"width_ratios": [1, 1.9]})
    th = np.linspace(0, 2 * np.pi, 300)
    ax1.plot(np.cos(th), np.sin(th), color=GRAY, lw=1)
    ax1.axhline(0, color="k", lw=0.5)
    ax1.axvline(0, color="k", lw=0.5)
    ax1.plot([p1, p2], [0, 0], "x", color=ORANGE, ms=12, mew=2.5, label="poles")
    ax1.plot([0], [0], "o", mfc="none", color=BLUE, ms=9, mew=1.8, label="zero")
    lim = max(1.3, 1.15 * rho)
    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)
    ax1.set_aspect("equal")
    ax1.set_xlabel("Re z")
    ax1.set_ylabel("Im z")
    ax1.set_title(r"$E(z) = z/((z-p_1)(z-p_2))$", fontsize=11)
    ax1.legend(fontsize=10, loc="upper left")

    if modes is not None:
        ax2.plot(k, modes[0], "s--", color=TEAL, ms=4, lw=1, label=r"mode $k_1 p_1^k$")
        ax2.plot(k, modes[1], "^--", color=GRAY, ms=4, lw=1, label=r"mode $k_2 p_2^k$")
    ml, sl, bl = ax2.stem(k, e_pf, label=r"$e_k$, partial fractions")
    plt.setp(ml, color=ORANGE, markersize=6)
    plt.setp(sl, color=ORANGE, lw=2)
    plt.setp(bl, color="k", lw=0.6)
    ax2.plot(k, e_div, "x", color=BLUE, ms=9, mew=1.6, label="long division")
    ax2.set_xlabel("k")
    ax2.set_ylabel(r"$e_k$")
    verdict = "decays" if rho < 1 - 1e-12 else ("grows" if rho > 1 + 1e-12 else "does not decay")
    ax2.set_title(f"$p_1$ = {p1:g}, $p_2$ = {p2:g}: $e_k$ {verdict}", fontsize=11)
    ax2.legend(fontsize=10, loc="best")
    fig.tight_layout()

    if repeated:
        print(f"repeated pole p = {p1:g}: E(z) = z/(z-p)^2  ->  e_k = k p^(k-1)")
    else:
        print(f"k1 = 1/(p1 - p2) = {k1:.4g},  k2 = 1/(p2 - p1) = {k2:.4g}  ->  "
              f"e_k = ({k1:.4g})({p1:g})^k + ({k2:.4g})({p2:g})^k")
    print("e_0..e_6 =", np.round(e_pf[:7], 4),
          "(long division agrees)" if np.allclose(e_pf, e_div, rtol=1e-9, atol=1e-9) else "")
    print(f"largest pole magnitude {rho:.3g}: e_k {verdict}")
    plt.show()


try:
    from ipywidgets import interact, FloatSlider
    interact(explore,
             p1=FloatSlider(value=1.0, min=-1.2, max=2.0, step=0.05, description="p1",
                            continuous_update=False),
             p2=FloatSlider(value=2.0, min=-1.2, max=2.0, step=0.05, description="p2",
                            continuous_update=False))
except ImportError:
    explore()
