r"""**How large can $T$ be?** A unit step goes through the zero-order hold into
$G_p(s) = 25/(s^2 + 2\zeta\cdot5\,s + 25)$, with $\zeta = 0.2$ by default: the true output
$c(t)$ peaks at 1.527 at $t = 0.64$ s. The samples $c(kT)$ come from the pulse transfer
function $G(z) = (1 - z^{-1})\,\mathcal{Z}[G_p(s)/s]$ and are exact, but they show only what
happens at $t = kT$. Start at $T = 1$ s (about 1.3 samples per oscillation, the peak is
missed), then shrink $T$ toward 0.1 s and watch the samples trace the response. Watch
the poles $z = e^{sT}$ move around the circle too.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.linalg import expm

plt.rcParams.update({"font.size": 11})
ORANGE, BLUE, TEAL, GRAY = "#E67E22", "#1D5FA8", "#1F6E78", "#6B7785"
WN = 5.0
T_END = 6.0


def _zoh(zeta, T):
    """ZOH equivalent of WN^2/(s^2 + 2 zeta WN s + WN^2): G(z) = num/den in powers of z."""
    A = np.array([[0.0, 1.0], [-WN ** 2, -2 * zeta * WN]])
    B = np.array([[0.0], [WN ** 2]])
    C = np.array([[1.0, 0.0]])
    M = np.zeros((3, 3))
    M[:2, :2], M[:2, 2:] = A, B
    E = expm(M * T)
    Phi, Gam = E[:2, :2], E[:2, 2:]
    num, den = signal.ss2tf(Phi, Gam, C, np.zeros((1, 1)))
    return num[0][1:], den          # num = b1 z + b2


def explore(T=1.0, zeta=0.2):
    T, zeta = round(T, 10), round(zeta, 10)
    num, den = _zoh(zeta, T)
    n = int(np.floor(T_END / T)) + 1
    c_k = signal.lfilter(np.r_[0.0, num], den, np.ones(n))      # unit step samples
    t = np.linspace(0, T_END, 2000)
    _, c_t = signal.step(([WN ** 2], [1.0, 2 * zeta * WN, WN ** 2]), T=t)
    poles = np.roots(den)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.0),
                                   gridspec_kw={"width_ratios": [1.9, 1]})
    ax1.plot(t, c_t, color=GRAY, lw=1.6, label="c(t)")
    ax1.plot(np.arange(n) * T, c_k, "o", color=ORANGE, ms=6, label="samples c(kT)")
    ax1.axhline(1, color="k", lw=0.6)
    ax1.set_xlim(0, T_END)
    ax1.set_ylim(0, max(1.7, c_t.max() + 0.15))
    ax1.set_xlabel("t (s)")
    ax1.set_ylabel("c")
    ax1.set_title(rf"Unit step through the ZOH: T = {T:g} s, $\zeta$ = {zeta:g}", fontsize=11)
    ax1.legend(fontsize=10, loc="lower right")

    th = np.linspace(0, 2 * np.pi, 300)
    ax2.plot(np.cos(th), np.sin(th), color=GRAY, lw=1)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.axvline(0, color="k", lw=0.5)
    ax2.plot(poles.real, poles.imag, "x", color=ORANGE, ms=12, mew=2.5,
             label=r"poles $e^{s_i T}$")
    ax2.set_xlim(-1.3, 1.3)
    ax2.set_ylim(-1.3, 1.3)
    ax2.set_aspect("equal")
    ax2.set_xlabel("Re z")
    ax2.set_ylabel("Im z")
    ax2.set_title("poles of G(z)", fontsize=11)
    ax2.legend(fontsize=10, loc="upper left")
    fig.tight_layout()

    sg = lambda v: f"{'-' if v < 0 else '+'} {abs(v):.4f}"
    print(f"G(z) = ({num[0]:.4f} z {sg(num[1])}) / (z^2 {sg(den[1])} z {sg(den[2])}),"
          f"  G(1) = {np.sum(num) / np.sum(den):.4f}")
    i = int(np.argmax(c_k))
    peak = (f"true peak {c_t.max():.3f} at t = {t[np.argmax(c_t)]:.2f} s" if c_t.max() > 1.001
            else "c(t) rises without overshoot")
    print(f"{peak};  largest sample c({i}) = {c_k[i]:.3f} at t = {i * T:g} s")
    if zeta < 1:
        wd = WN * np.sqrt(1 - zeta ** 2)
        print(f"oscillation period 2 pi/wd = {2 * np.pi / wd:.2f} s: "
              f"{2 * np.pi / (wd * T):.1f} samples per period")
    else:
        print("no oscillation (zeta >= 1)")
    plt.show()


try:
    from ipywidgets import interact, FloatSlider
    interact(explore,
             T=FloatSlider(value=1.0, min=0.05, max=1.5, step=0.05, description="T (s)",
                           continuous_update=False),
             zeta=FloatSlider(value=0.2, min=0.05, max=1.0, step=0.05, description="zeta",
                              continuous_update=False))
except ImportError:
    explore()
