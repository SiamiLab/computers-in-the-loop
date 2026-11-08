r"""**Ride the root locus.** The running loop: a zero-order hold, $G_p(s) = 1/(s(s+1))$, gain $K$
and unity feedback. The gray curve is the root locus of $1 + K\,G(z) = 0$; the orange crosses are
the closed-loop poles at your $K$. At $T = 1$ s the poles break away from the real axis at
$K = 0.196$, cross the dashed $\zeta = 0.5$ curve at $K = 0.516$, and leave the unit circle at
$K = 2.39$. Shrink $T$ and the critical gain grows (about $2/T + 1/3$). Switch to $D(z) = K/z$ to
see what one sample of computation delay does: the limit drops to 0.805.
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11})
ORANGE, BLUE, TEAL, GRAY = "#E67E22", "#1D5FA8", "#1F6E78", "#6B7785"
NO_DELAY, DELAY = "D(z) = K", "D(z) = K/z"


def _loop(T, controller):
    """Numerator and denominator of the open loop D(z)G(z)/K, descending powers of z."""
    p = np.exp(-T)
    num = np.array([T - 1 + p, 1 - p - T * p])
    den = np.array([1.0, -(1 + p), p])
    if controller == DELAY:
        den = np.polymul(den, [1.0, 0.0])
    return num, den


def _poles(K, num, den):
    return np.roots(np.polyadd(den, K * num))


def explore(K=1.0, T=1.0, controller=NO_DELAY):
    num, den = _loop(T, controller)
    poles = _poles(K, num, den)
    rmax = np.max(np.abs(poles))

    lo, hi = 1e-6, 10.0 / T + 5.0                      # critical gain by bisection on max |z|
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if np.max(np.abs(_poles(mid, num, den))) < 1 else (lo, mid)
    k_cr = lo

    Ks = np.geomspace(1e-3, 60.0 / T, 900)
    locus = np.concatenate([_poles(k, num, den) for k in Ks])

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color="k", lw=0.9)
    ph = np.linspace(0, np.pi, 300)                    # zeta = 0.5: z = exp(-zeta/sqrt(1-zeta^2) theta + j theta)
    rr = np.exp(-ph / np.sqrt(3))
    for sgn in (1, -1):
        ax.plot(rr * np.cos(ph), sgn * rr * np.sin(ph), "--", color=TEAL, lw=1.2,
                label="$\\zeta = 0.5$" if sgn == 1 else None)
    ax.plot(locus.real, locus.imag, ".", color=GRAY, ms=2.2)
    op = np.roots(den)
    ax.plot(op.real, op.imag, "x", color=BLUE, ms=10, mew=2, label="open-loop poles")
    z0 = -num[1] / num[0]
    ax.plot([z0], [0], "o", mfc="white", mec=BLUE, mew=2, ms=9, label="zero")
    ax.plot(poles.real, poles.imag, "X", color=ORANGE, ms=13, mec="k", mew=0.8,
            label=f"closed-loop poles, K = {K:g}")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlim(-2.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    ax.set_xlabel("Re z")
    ax.set_ylabel("Im z")
    ax.set_title(f"{controller}, T = {T:g} s: critical gain {k_cr:.3f}", fontsize=11)
    ax.legend(fontsize=10, loc="lower left")
    fig.tight_layout()

    print("closed-loop poles:", ", ".join(f"{z.real:.3f}{z.imag:+.3f}j" for z in poles),
          f"   max |z| = {rmax:.3f}  ->  " + ("stable" if rmax < 1 else "NOT stable"))
    zp = poles[np.argmax(poles.imag)]
    if abs(zp.imag) > 1e-9 and abs(zp) < 1:
        s = np.log(zp) / T
        print(f"complex pair: zeta = {-s.real / abs(s):.3f}, wn = {abs(s):.3f} rad/s")
    print(f"critical gain at T = {T:g} s: K = {k_cr:.3f};  zero of G(z) at z = {z0:.3f}")
    plt.show()


try:
    from ipywidgets import interact, FloatSlider, Dropdown
    interact(explore,
             K=FloatSlider(value=1.0, min=0.02, max=10.0, step=0.02, description="K",
                           continuous_update=False),
             T=FloatSlider(value=1.0, min=0.1, max=2.0, step=0.05, description="T (s)",
                           continuous_update=False),
             controller=Dropdown(options=[NO_DELAY, DELAY], value=NO_DELAY, description="D(z)"))
except ImportError:
    explore()
