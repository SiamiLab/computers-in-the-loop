r"""**The robot-arm position loop.** The loop is $\Theta_L/\Theta_c = GG_c/(1+GG_c)$, the
one-loop case of Mason's formula. With the inductance neglected, $G(s) = 1/(s(s+6))$ and
$G_c = k$ give $k/(s^2 + 6s + k)$. Move $k$: at $k = 9$ the two poles meet at $-3$, at
$k = 25$ they are $-3 \pm j4$. Then raise the inductance $L$ and push $k$ up: the
third-order loop, which keeps $L$, eventually goes unstable, while the second-order model
never does.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

plt.rcParams.update({"font.size": 11})
ORANGE, BLUE, TEAL, GRAY = "#E67E22", "#1D5FA8", "#1F6E78", "#6B7785"

# reflected to the motor shaft (SI units)
J = B = 0.05
R, KT, KM, N = 1.0, 0.5, 0.5, 0.1


def _overshoot(y):
    """Percent overshoot over the final value 1 (both loops have DC gain 1)."""
    return max(0.0, 100 * (y.max() - 1))


def explore(k=25.0, L_mH=10.0):
    L = L_mH / 1000
    # L neglected: k/(s^2 + 6s + k)
    den2 = np.array([R * J, R * B + KT * KM, N * KT * k]) / (R * J)
    num2 = [N * KT * k / (R * J)]
    # L kept: n Kt k / (s[(R + Ls)(Js + B) + Kt Km] + n Kt k)
    motor = np.polyadd(np.polymul([L, R], [J, B]), [KT * KM])
    den3 = np.polyadd(np.polymul(motor, [1, 0]), [N * KT * k])
    den3 = np.trim_zeros(den3, "f")
    num3 = [N * KT * k]
    p2, p3 = np.roots(den2), np.roots(den3)
    stable3 = np.all(p3.real < -1e-12)

    slow = np.min(np.abs(p2.real))
    t_end = float(np.clip(8 / slow, 2.0, 30.0))
    t = np.linspace(0, t_end, 1500)
    _, y2 = signal.step((num2, den2), T=t)
    _, y3 = signal.step((num3, den3), T=t)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0),
                                   gridspec_kw={"width_ratios": [1.8, 1]})
    ax1.plot(t, y2, color=ORANGE, lw=2, label="L neglected (second order)")
    ax1.plot(t, y3, "--", color=BLUE, lw=1.6, label=f"L = {L_mH:g} mH kept (third order)")
    ax1.axhline(1, color="k", lw=0.6)
    hi = max(1.8, y2.max() + 0.1)
    ax1.set_ylim(-0.3 if stable3 else -1.5, hi if stable3 else 3.0)
    ax1.set_xlim(0, t_end)
    ax1.set_xlabel("t (s)")
    ax1.set_ylabel(r"arm angle $\theta_L$ / command")
    ax1.set_title(f"Unit step command, $G_c = k = {k:g}$", fontsize=11)
    ax1.legend(fontsize=10, loc="upper right")

    ax2.axhline(0, color="k", lw=0.5)
    ax2.axvline(0, color="k", lw=0.5)
    ax2.plot(p2.real, p2.imag, "x", color=ORANGE, ms=11, mew=2.5, label="L neglected")
    ax2.plot(p3.real, p3.imag, "o", mfc="none", color=BLUE, ms=10, mew=1.8, label="L kept")
    ax2.set_xlim(-12, 3)
    ymax = max(6.0, 1.2 * np.max(np.abs(np.r_[p2.imag, p3.imag])))
    ax2.set_ylim(-ymax, ymax)
    ax2.set_xlabel("Re s (1/s)")
    ax2.set_ylabel("Im s (rad/s)")
    ax2.set_title("closed-loop poles (fast pole may be off scale)", fontsize=10)
    ax2.legend(fontsize=10, loc="upper left")
    fig.tight_layout()

    fmt = lambda p: ", ".join(f"{z.real:.2f}{z.imag:+.2f}j" for z in p)
    wn = np.sqrt(k)
    print(f"L neglected: k/(s^2 + 6s + k), poles {fmt(p2)}; wn = {wn:.2f} rad/s, "
          f"zeta = {3 / wn:.2f}, overshoot {_overshoot(y2):.1f}%")
    if stable3:
        print(f"L = {L_mH:g} mH kept: poles {fmt(p3)}, overshoot {_overshoot(y3):.1f}%")
    else:
        print(f"L = {L_mH:g} mH kept: poles {fmt(p3)}: UNSTABLE")
    if L > 0:
        # Routh on L J s^3 + (R J + L B) s^2 + (R B + Kt Km) s + n Kt k
        kmax = (R * J + L * B) * (R * B + KT * KM) / (L * J * N * KT)
        print(f"third-order loop is stable for 0 < k < {kmax:.0f}; the second-order model for every k > 0")
    plt.show()


try:
    from ipywidgets import interact, FloatSlider
    interact(explore,
             k=FloatSlider(value=25.0, min=1.0, max=200.0, step=1.0, description="k = G_c",
                           continuous_update=False),
             L_mH=FloatSlider(value=10.0, min=0.0, max=50.0, step=1.0, description="L (mH)",
                              continuous_update=False))
except ImportError:
    explore()
