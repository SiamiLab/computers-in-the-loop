r"""**One analog lead, two digital versions.** The lead $G_c(s) = 4(s+1)/(s+2)$ around
$G_p(s) = 1/(s(s+1))$ gives the analog loop $4/(s^2 + 2s + 4)$: poles $-1 \pm j\sqrt3$,
16.3% overshoot. Here the computer runs it as $D(z)$, behind a zero-order hold. Tustin's rule is
always shown (orange); pick a second method to compare (blue). At $T = 0.2$ s every method is
close (Tustin: 25.7% overshoot of the samples). Push $T$ toward 1 s and watch the hold's lag eat
the phase margin: the samples of the Tustin loop overshoot by 82%.
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11})
ORANGE, BLUE, TEAL, GRAY = "#E67E22", "#1D5FA8", "#1F6E78", "#6B7785"
METHODS = ["ZOH equivalent", "matched pole-zero", "backward Euler", "forward Euler"]
T_SHOW, T_LONG = 14.0, 40.0


def _controller(method, T):
    """D(z) = (d1 z + d0)/(z - e) for Gc(s) = 4(s+1)/(s+2)."""
    if method == "Tustin":                             # s = (2/T)(z-1)/(z+1)
        c = 2.0 / T
        num, den = 4 * np.array([c + 1, 1 - c]), np.array([c + 2, 2 - c])
    elif method == "ZOH equivalent":                   # 4 - 2/(s+2), step invariant
        q = np.exp(-2 * T)
        num, den = np.array([4.0, -2 - 2 * q]), np.array([1.0, -q])
    elif method == "matched pole-zero":                # z = e^{sT}, DC gain 2
        num, den = 2 * (1 + np.exp(-T)) * np.array([1.0, -np.exp(-T)]), np.array([1.0, -np.exp(-2 * T)])
    elif method == "backward Euler":                   # s = (z-1)/(Tz)
        num, den = 4 * np.array([1 + T, -1.0]), np.array([1 + 2 * T, -1.0])
    else:                                              # forward Euler, s = (z-1)/T
        num, den = 4 * np.array([1.0, T - 1]), np.array([1.0, 2 * T - 1])
    return num / den[0], den / den[0]


def _simulate(num, den, T):
    """Unit step through the loop; the plant 1/(s(s+1)) is exact between the samples."""
    p = np.exp(-T)
    b1 = T - 1 + p
    n = int(np.ceil(T_LONG / T))
    tau = np.linspace(0, T, 11)[:-1]
    x1 = x2 = 0.0
    u_old = e_old = 0.0
    ck, tf, cf = np.zeros(n + 1), [], []
    for k in range(n + 1):
        ck[k] = x1
        e = 1.0 - x1
        u = -den[1] * u_old + num[0] * e + num[1] * e_old
        u_old, e_old = u, e
        if k * T <= T_SHOW:
            tf.append(k * T + tau)
            cf.append(x1 + x2 * (1 - np.exp(-tau)) + u * (tau - 1 + np.exp(-tau)))
        x1, x2 = x1 + x2 * (1 - p) + u * b1, x2 * p + u * (1 - p)
    return ck, np.concatenate(tf), np.concatenate(cf)


def _closed_loop_poles(num, den, T):
    p = np.exp(-T)
    gn, gd = np.array([T - 1 + p, 1 - p - T * p]), np.array([1.0, -(1 + p), p])
    return np.roots(np.polyadd(np.polymul(den, gd), np.polymul(num, gn)))


def explore(T=0.2, compare="ZOH equivalent"):
    t = np.linspace(0, T_SHOW, 600)
    c_analog = 1 - np.exp(-t) * (np.cos(np.sqrt(3) * t) + np.sin(np.sqrt(3) * t) / np.sqrt(3))
    target = np.exp(T * np.array([-1 + 1j * np.sqrt(3), -1 - 1j * np.sqrt(3)]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [2.1, 1]})
    ax1.plot(t, c_analog, "--", color=GRAY, lw=1.5, label="analog loop")
    th = np.linspace(0, 2 * np.pi, 300)
    ax2.plot(np.cos(th), np.sin(th), color=GRAY, lw=1)
    ax2.plot(target.real, target.imag, "+", color=GRAY, ms=14, mew=2, label="$e^{sT}$ of analog poles")
    lines = []
    worst = 0.0
    for method, col, mk in [(compare, BLUE, "s"), ("Tustin", ORANGE, "o")]:
        num, den = _controller(method, T)
        ck, tf, cf = _simulate(num, den, T)
        poles = _closed_loop_poles(num, den, T)
        rmax = np.max(np.abs(poles))
        worst = max(worst, rmax)
        ks = np.arange(len(ck))
        keep = ks * T <= T_SHOW
        ax1.plot(tf, cf, color=col, lw=1.4)
        ax1.plot(ks[keep] * T, ck[keep], mk, color=col, ms=5 if T >= 0.3 else 3, label=method)
        ax2.plot(poles.real, poles.imag, "x", color=col, ms=10, mew=2.2, label=method)
        os_txt = f"{100 * (ck.max() - 1):.1f}%" if rmax < 1 else "unstable"
        lines.append(f"{method:18s} D(z) = {num[0]:.4f}(z {num[1] / num[0]:+.4f})/(z {den[1]:+.4f});"
                     f"  max |pole| = {rmax:.3f};  overshoot of the samples {os_txt}")
    ax1.axhline(1, color="k", lw=0.6)
    ax1.set_xlim(0, T_SHOW)
    ax1.set_ylim(-0.1, 2.2)
    ax1.set_xlabel("t (s)")
    ax1.set_ylabel("c")
    ax1.set_title(f"unit step with the lead run at T = {T:g} s", fontsize=11)
    ax1.legend(fontsize=10, loc="lower right")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.axvline(0, color="k", lw=0.5)
    lim = max(1.2, 1.05 * worst)
    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)
    ax2.set_aspect("equal")
    ax2.set_xlabel("Re z")
    ax2.set_ylabel("Im z")
    ax2.set_title("closed-loop poles", fontsize=11)
    ax2.legend(fontsize=10, loc="lower left")
    fig.tight_layout()

    print(f"analog: 16.3% overshoot;  its poles -1 +- j1.732 map to {target[0].real:.4f} +- j{target[0].imag:.4f}")
    for line in reversed(lines):
        print(line)
    plt.show()


try:
    from ipywidgets import interact, FloatSlider, Dropdown
    interact(explore,
             T=FloatSlider(value=0.2, min=0.05, max=1.0, step=0.05, description="T (s)",
                           continuous_update=False),
             compare=Dropdown(options=METHODS, value="ZOH equivalent", description="compare"))
except ImportError:
    explore()
