r"""**The pole of $a^k$ and where its series converges.** The sequence $e_k = a^k$ has
$E(z) = \sum_k a^k z^{-k} = z/(z-a)$, but only for $|z| > |a|$, the exterior of the disk
through the pole. Move $a$ and watch the sequence: it decays when the pole is inside the unit
circle and grows when it is outside. Then move the test point $z$ inside the disk: the closed
form still returns a number (try $a = 0.5$, $z = 0.25$: it says $-1$), but the partial sums
run away.
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11})
ORANGE, BLUE, TEAL, GRAY = "#E67E22", "#1D5FA8", "#1F6E78", "#6B7785"
NK = 21        # samples shown, k = 0..20


def explore(a=0.5, z=1.0):
    a, z = round(a, 10), round(z, 10)        # slider values such as 0.49999999999
    k = np.arange(NK)
    e = a ** k
    on_pole = abs(z - a) < 1e-12
    inside_roc = abs(z) > abs(a)
    closed = np.inf if on_pole else z / (z - a)
    partial = np.cumsum(e * z ** (-k.astype(float)))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.5, 3.9),
                                        gridspec_kw={"width_ratios": [1.25, 1, 1.25]})
    ml, sl, bl = ax1.stem(k, e)
    plt.setp(ml, color=ORANGE, markersize=5)
    plt.setp(sl, color=ORANGE)
    plt.setp(bl, color="k", lw=0.6)
    ax1.set_xlabel("k")
    ax1.set_ylabel(r"$e_k = a^k$")
    ax1.set_title(f"a = {a:g}: " + ("decays" if abs(a) < 1 else
                                    "stays bounded" if abs(a) == 1 else "grows"), fontsize=11)

    lim = 1.6
    th = np.linspace(0, 2 * np.pi, 300)
    ax2.add_patch(plt.Rectangle((-lim, -lim), 2 * lim, 2 * lim, color=TEAL, alpha=0.15, lw=0))
    ax2.add_patch(plt.Circle((0, 0), abs(a), color="white", lw=0))
    ax2.plot(abs(a) * np.cos(th), abs(a) * np.sin(th), color=TEAL, lw=1.2)
    ax2.plot(np.cos(th), np.sin(th), color=GRAY, lw=1, ls="--")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.axvline(0, color="k", lw=0.5)
    ax2.plot([a], [0], "x", color=ORANGE, ms=12, mew=2.5, label="pole z = a")
    ax2.plot([z], [0], "o", color=BLUE, ms=8, label="test point z")
    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)
    ax2.set_aspect("equal")
    ax2.set_xlabel("Re z")
    ax2.set_ylabel("Im z")
    ax2.set_title("ROC |z| > |a| (shaded)", fontsize=11)
    ax2.legend(fontsize=10, loc="upper left", framealpha=0.9)

    ax3.plot(k, partial, "o-", color=BLUE, ms=4, lw=1.2, label="partial sums")
    if np.isfinite(closed):
        ax3.axhline(closed, color=ORANGE, ls="--", lw=2, label=f"z/(z-a) = {closed:.3g}")
    ax3.set_xlabel("k (last term in the sum)")
    ax3.set_ylabel(r"$\sum_{i\leq k} a^i z^{-i}$")
    ax3.set_title(f"series at z = {z:g}: " + ("converges" if inside_roc else "diverges"),
                  fontsize=11)
    if not inside_roc:                      # the sums leave the window: that is the point
        c = closed if np.isfinite(closed) else 0.0
        ax3.set_ylim(c - 6, c + 6)
    ax3.legend(fontsize=10, loc="upper left")
    fig.tight_layout()

    print(f"pole z = a = {a:g}; ROC |z| > {abs(a):g}; "
          + ("pole inside the unit circle: a^k -> 0" if abs(a) < 1 else
             "pole on the unit circle: a^k does not decay" if abs(a) == 1 else
             "pole outside the unit circle: a^k grows"))
    if on_pole:
        print(f"z = {z:g} sits on the pole: z/(z-a) is not defined")
    else:
        print(f"closed form z/(z-a) at z = {z:g}: {closed:.4f}")
    print(f"partial sum with {NK} terms: {partial[-1]:.4g}  ("
          + ("inside the ROC, it approaches the closed form)" if inside_roc else
             "outside the ROC, the series has no sum)"))
    plt.show()


try:
    from ipywidgets import interact, FloatSlider
    interact(explore,
             a=FloatSlider(value=0.5, min=-1.2, max=1.2, step=0.05, description="pole a",
                           continuous_update=False),
             z=FloatSlider(value=1.0, min=0.1, max=1.6, step=0.05, description="test z",
                           continuous_update=False))
except ImportError:
    explore()
