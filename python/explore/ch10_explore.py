r"""**The running loop: gain and sampling period.** A zero-order hold drives
$G_p(s) = 1/(s(s+1))$, with $D(z) = K$ and unity feedback. Start at $K = 1$, $T = 1$ s: the
poles sit at $0.5 \pm j0.618$ and the samples peak at 1.40, while the output between the samples
peaks at 1.45. Raise $K$ toward 2.39, or $T$ toward 3.92 s, and watch the poles reach the unit
circle and the response ring, then grow. Small $T$ brings the samples back to the analog loop.
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11})
ORANGE, BLUE, TEAL, GRAY = "#E67E22", "#1D5FA8", "#1F6E78", "#6B7785"
T_END = 20.0


def _analog_step(K, t):
    """Step response of K/(s^2 + s + K) (the loop without the sampler)."""
    if abs(K - 0.25) < 1e-9:                           # double root: nudge it apart
        K += 1e-7
    s1, s2 = np.roots([1.0, 1.0, K]).astype(complex)
    return np.real(1 + (s2 * np.exp(s1 * t) - s1 * np.exp(s2 * t)) / (s1 - s2))


def explore(K=1.0, T=1.0):
    p = np.exp(-T)
    b1, b0 = T - 1 + p, 1 - p - T * p                  # G(z) = (b1 z + b0)/((z - 1)(z - p))
    poles = np.roots([1.0, K * b1 - (1 + p), p + K * b0])
    rmax = np.max(np.abs(poles))
    stable = rmax < 1 - 1e-12

    # the loop, exact between the samples: states c and dc/dt, input m held over each period
    n = int(np.ceil(T_END / T))
    tau = np.linspace(0, T, 21)[:-1]
    x1 = x2 = 0.0
    ck, tf, cf = [], [], []
    for k in range(n + 1):
        ck.append(x1)
        m = K * (1.0 - x1)
        tf.append(k * T + tau)
        cf.append(x1 + x2 * (1 - np.exp(-tau)) + m * (tau - 1 + np.exp(-tau)))
        x1, x2 = x1 + x2 * (1 - p) + m * b1, x2 * p + m * (1 - p)
    ck, tf, cf = np.array(ck), np.concatenate(tf), np.concatenate(cf)

    t = np.linspace(0, T_END, 800)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [2.1, 1]})
    ax1.plot(t, _analog_step(K, t), "--", color=GRAY, lw=1.4, label="analog loop")
    ax1.plot(tf, cf, color=ORANGE, lw=1.5, label="c(t), sampled loop")
    ax1.plot(np.arange(n + 1) * T, ck, "o", color=ORANGE, ms=5 if T >= 0.3 else 2.5,
             label="samples c(kT)")
    ax1.axhline(1, color="k", lw=0.6)
    ax1.set_xlim(0, T_END)
    if stable:
        ax1.set_ylim(min(-0.2, cf.min() - 0.1), max(2.2, cf.max() + 0.1))
    else:
        ax1.set_ylim(-2.0, 4.0)
    ax1.set_xlabel("t (s)")
    ax1.set_ylabel("c")
    ax1.set_title(f"unit step, K = {K:g}, T = {T:g} s" + ("" if stable else ": unstable"), fontsize=11)
    ax1.legend(fontsize=10, loc="lower right")

    th = np.linspace(0, 2 * np.pi, 300)
    ax2.plot(np.cos(th), np.sin(th), color=GRAY, lw=1)
    ax2.plot(poles.real, poles.imag, "x", color=ORANGE, ms=11, mew=2.5)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.axvline(0, color="k", lw=0.5)
    lim = max(1.3, 1.1 * rmax)
    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)
    ax2.set_aspect("equal")
    ax2.set_xlabel("Re z")
    ax2.set_ylabel("Im z")
    ax2.set_title("closed-loop poles", fontsize=11)
    fig.tight_layout()

    print(f"G(z) = ({b1:.4f} z + {b0:.4f}) / ((z - 1)(z - {p:.4f}));  "
          f"1 + KG = 0: z^2 {K * b1 - (1 + p):+.4f} z {p + K * b0:+.4f}")
    print("closed-loop poles:", ", ".join(f"{z.real:.3f}{z.imag:+.3f}j" for z in poles),
          f"   max |z| = {rmax:.3f}")
    if stable:
        line = f"stable: samples peak at {ck.max():.2f}, c(t) peaks at {cf.max():.2f}"
        zp = poles[np.argmax(poles.imag)]
        if abs(zp.imag) > 1e-9:
            s = np.log(zp) / T
            line += f";  zeta = {-s.real / abs(s):.2f}, wn = {abs(s):.2f} rad/s"
        print(line)
    else:
        print("not stable: a pole is on or outside the unit circle")
    plt.show()


try:
    from ipywidgets import interact, FloatSlider
    interact(explore,
             K=FloatSlider(value=1.0, min=0.1, max=3.0, step=0.05, description="K",
                           continuous_update=False),
             T=FloatSlider(value=1.0, min=0.1, max=4.0, step=0.05, description="T (s)",
                           continuous_update=False))
except ImportError:
    explore()
