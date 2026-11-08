r"""**Where is the running loop stable?** A zero-order hold drives $G_p(s) = 1/(s(s+1))$, with
$D(z) = K$ (or $K/z$, one sample of computation delay) and unity feedback. The left panel is the
stable region in the $(T, K)$ plane, from Jury's test; the orange dot is your choice. At $T = 1$ s
the loop is stable for $0 < K < 2.39$, and only for $0 < K < 0.805$ with the delay. On the right,
the second-order test: the point $(a_1, a_0)$ of $z^2 + a_1 z + a_0$ must stay inside the triangle.
Shrink $T$ and watch the gain limit grow, roughly like $2/T$.
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11})
ORANGE, BLUE, TEAL, GRAY = "#E67E22", "#1D5FA8", "#1F6E78", "#6B7785"
NO_DELAY, DELAY = "D(z) = K", "D(z) = K/z"


def _coefficients(K, T, controller):
    """Characteristic polynomial of 1 + D G = 0, descending powers of z."""
    p = np.exp(-T)
    b1, b0 = T - 1 + p, 1 - p - T * p                  # G(z) = (b1 z + b0)/((z - 1)(z - p))
    if controller == NO_DELAY:
        return np.array([1.0, K * b1 - (1 + p), p + K * b0])
    return np.array([1.0, -(1 + p), p + K * b1, K * b0])


def _jury(c):
    """Jury's conditions for a monic polynomial of degree 2 or 3: (values, all hold?)."""
    q1, qm1 = np.polyval(c, 1.0), (-1) ** (len(c) - 1) * np.polyval(c, -1.0)
    a0 = c[-1]
    cond = [q1 > 0, qm1 > 0, abs(a0) < 1]
    if len(c) == 4:                                    # z^3 + a2 z^2 + a1 z + a0
        cond.append(abs(a0 ** 2 - 1) > abs(a0 * c[1] - c[2]))
    return (q1, qm1, a0), all(cond)


def _k_limit(T, controller):
    """Largest K with every condition of Jury's test met (bisection; stable for 0 < K < limit)."""
    lo, hi = 1e-6, 4.0 / T + 5.0
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if _jury(_coefficients(mid, T, controller))[1] else (lo, mid)
    return lo


def explore(K=1.0, T=1.0, controller=NO_DELAY):
    c = _coefficients(K, T, controller)
    (q1, qm1, a0), stable = _jury(c)
    roots = np.roots(c)
    k_lim = _k_limit(T, controller)

    Ts = np.linspace(0.05, 3.0, 60)
    Ks = np.array([_k_limit(t, controller) for t in Ts])
    k_top = max(6.0, 1.2 * K)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    ax1.fill_between(Ts, 0, np.minimum(Ks, k_top), color=TEAL, alpha=0.18, label="stable")
    ax1.plot(Ts, Ks, color=TEAL, lw=2, label="gain limit")
    if controller == DELAY:
        ax1.plot(Ts, [_k_limit(t, NO_DELAY) for t in Ts], "--", color=GRAY, lw=1.4,
                 label="limit without the delay")
    ax1.plot([T], [K], "o" if stable else "X", color=ORANGE, ms=11, mec="k",
             label=f"K = {K:g}, T = {T:g} s")
    ax1.set_xlim(0, 3.0)
    ax1.set_ylim(0, k_top)
    ax1.set_xlabel("T (s)")
    ax1.set_ylabel("K")
    ax1.set_title(f"{controller}: stable region from Jury's test", fontsize=11)
    ax1.legend(fontsize=10, loc="upper right")

    if controller == NO_DELAY:
        tri = np.array([[-2, 1], [2, 1], [0, -1], [-2, 1]])
        ax2.fill(tri[:, 0], tri[:, 1], color=TEAL, alpha=0.18)
        ax2.plot(tri[:, 0], tri[:, 1], color=TEAL, lw=1.5)
        kk = np.linspace(0, max(k_lim, K) * 1.3, 60)
        path = np.array([_coefficients(k, T, controller) for k in kk])
        ax2.plot(path[:, 1], path[:, 2], color=GRAY, lw=1.2, label="K from 0 up")
        ax2.plot([c[1]], [c[2]], "o" if stable else "X", color=ORANGE, ms=11, mec="k",
                 label="($a_1$, $a_0$) now")
        ax2.set_xlim(-2.6, 2.6)
        ax2.set_ylim(-1.4, 1.9)
        ax2.set_xlabel("$a_1$")
        ax2.set_ylabel("$a_0$")
        ax2.set_title("$|a_0| < 1$, $Q(1) > 0$, $Q(-1) > 0$", fontsize=11)
        ax2.legend(fontsize=10, loc="upper left")
    else:
        th = np.linspace(0, 2 * np.pi, 300)
        ax2.plot(np.cos(th), np.sin(th), color=GRAY, lw=1)
        ax2.plot(roots.real, roots.imag, "x", color=ORANGE, ms=11, mew=2.5)
        ax2.axhline(0, color="k", lw=0.5)
        ax2.axvline(0, color="k", lw=0.5)
        lim = max(1.3, 1.1 * np.max(np.abs(roots)))
        ax2.set_xlim(-lim, lim)
        ax2.set_ylim(-lim, lim)
        ax2.set_aspect("equal")
        ax2.set_xlabel("Re z")
        ax2.set_ylabel("Im z")
        ax2.set_title("closed-loop poles (cubic)", fontsize=11)
    fig.tight_layout()

    powers = ["z^3", "z^2", "z", ""][4 - len(c):]
    print("Q(z) = " + " ".join(f"{v:+.4f} {w}" for v, w in zip(c, powers)).strip())
    print(f"Jury: Q(1) = {q1:.4f}, (-1)^n Q(-1) = {qm1:.4f}, |a0| = {abs(a0):.4f}"
          + ("" if len(c) == 3 else f", |a0^2 - 1| = {abs(a0 ** 2 - 1):.4f} vs |a0 a2 - a1| = {abs(a0 * c[1] - c[2]):.4f}"))
    if len(c) == 3:
        print(f"bilinear: Q_w(w) = {qm1:.4f} w^2 {2 * (1 - a0):+.4f} w {q1:+.4f}  (all > 0 means stable)")
    print(("stable" if stable else "NOT stable") + f";  largest |z| = {np.max(np.abs(roots)):.3f};"
          f"  at T = {T:g} s the loop is stable for 0 < K < {k_lim:.3f}")
    plt.show()


try:
    from ipywidgets import interact, FloatSlider, Dropdown
    interact(explore,
             K=FloatSlider(value=1.0, min=0.05, max=5.0, step=0.05, description="K",
                           continuous_update=False),
             T=FloatSlider(value=1.0, min=0.1, max=3.0, step=0.05, description="T (s)",
                           continuous_update=False),
             controller=Dropdown(options=[NO_DELAY, DELAY], value=NO_DELAY, description="D(z)"))
except ImportError:
    explore()
