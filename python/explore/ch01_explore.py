r"""**The laser loop on a computer.** The analog design $K/s$ around $1/(s+a)$, $a = 2$ s$^{-1}$,
is stable for every $K$. Here the same integrator runs on a computer as
$u_k = u_{k-1} + KT\,e_{k-1}$, with a zero-order hold in front of the plant.
Keep $K = 1$ and push $T$ from 0.1 s toward 2 s: the overshoot grows, and past $T = 2$ s the
output oscillates, then grows. Then change $K$ and watch the limit move to $T = a/K$.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

plt.rcParams.update({"font.size": 11})
ORANGE, BLUE, TEAL, GRAY = "#E67E22", "#1D5FA8", "#1F6E78", "#6B7785"

A = 2.0          # plant 1/(s + a), a = 2 1/s
T_END = 20.0     # seconds shown


def _digital_loop(T, K, r, d, n):
    """Samples of the sampled loop and the held control, from rest."""
    p = np.exp(-A * T)
    b = (1 - p) / A
    y = np.zeros(n + 1)
    u = np.zeros(n + 1)
    e = np.zeros(n + 1)
    for k in range(n):
        e[k] = r - y[k]
        if k >= 1:
            u[k] = u[k - 1] + K * T * e[k - 1]       # rectangle rule, error one sample old
        y[k + 1] = p * y[k] + b * (u[k] + d)         # ZOH + plant, exact at the samples
    return y, u


def explore(T=1.0, K=1.0, input="step command r"):
    disturbance = input.startswith("step disturbance")
    r, d = (0.0, 1.0) if disturbance else (1.0, 0.0)
    n = int(np.ceil(T_END / T))
    y, u = _digital_loop(T, K, r, d, n)

    # between samples the plant sees a constant input: exact exponential arcs
    tau = np.linspace(0, T, 25)
    t_c = np.concatenate([k * T + tau for k in range(n)])
    y_c = np.concatenate([np.exp(-A * tau) * y[k] + (1 - np.exp(-A * tau)) * (u[k] + d) / A
                          for k in range(n)])

    # analog design: Y/R = K/(s^2 + a s + K), Y/D = s/(s^2 + a s + K)
    t_a = np.linspace(0, T_END, 801)
    num = [1.0, 0.0] if disturbance else [K]
    _, y_a = signal.step(([*num], [1.0, A, K]), T=t_a)

    # modes of the sampled loop: lambda^2 - (1 + p) lambda + (p + b K T) = 0
    p = np.exp(-A * T)
    b = (1 - p) / A
    lam = np.roots([1.0, -(1 + p), p + b * K * T])
    rho = np.max(np.abs(lam))
    stable = rho < 1 - 1e-12

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0),
                                   gridspec_kw={"width_ratios": [2.1, 1]})
    ax1.plot(t_a, y_a, "--", color=GRAY, lw=1.5, label="analog design")
    ax1.plot(t_c, y_c, color=ORANGE, lw=1.6, label="y(t), computer in the loop")
    k_show = np.arange(n + 1)
    ax1.plot(k_show * T, y, "o", color=ORANGE, ms=4 if T >= 0.3 else 2, label="samples y(kT)")
    ax1.axhline(0 if disturbance else 1, color="k", lw=0.6)
    ax1.set_xlim(0, T_END)
    lo, hi = min(-0.2, y_a.min() - 0.1), max(1.9 if not disturbance else 0.8, y_a.max() + 0.1)
    if stable:
        lo, hi = min(lo, y.min() - 0.1), max(hi, y.max() + 0.1)
    else:
        lo, hi = -3.0, 4.0
    ax1.set_ylim(lo, hi)
    ax1.set_xlabel("t (s)")
    ax1.set_ylabel("beam position y")
    what = "unit step disturbance d (r = 0)" if disturbance else "unit step command r"
    ax1.set_title(f"T = {T:g} s, K = {K:g}: {what}", fontsize=11)
    ax1.legend(fontsize=10, loc="upper right")

    th = np.linspace(0, 2 * np.pi, 300)
    ax2.plot(np.cos(th), np.sin(th), color=GRAY, lw=1)
    ax2.plot(lam.real, lam.imag, "x", color=ORANGE, ms=11, mew=2.5)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.axvline(0, color="k", lw=0.5)
    lim = max(1.3, 1.1 * rho)
    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)
    ax2.set_aspect("equal")
    ax2.set_xlabel("Re $\\lambda$")
    ax2.set_ylabel("Im $\\lambda$")
    ax2.set_title("modes $\\lambda$ and the unit circle", fontsize=11)
    fig.tight_layout()

    print(f"p = e^(-aT) = {p:.4f},  b = (1 - p)/a = {b:.4f}")
    print("modes lambda =", ", ".join(f"{z.real:.3f}{z.imag:+.3f}j" for z in lam),
          f"   max |lambda| = {rho:.3f}")
    if stable:
        peak = y.max()
        label = "disturbance peak" if disturbance else "step peak"
        print(f"stable: {label} of y = {peak:.3f},  decay per second = {-np.log(rho) / T:.3f}")
    else:
        print("not stable: the samples do not decay")
    print(f"sampled loop is stable exactly when T < a/K = {A / K:.2f} s "
          "(the analog loop: every K > 0)")
    plt.show()


try:
    from ipywidgets import interact, FloatSlider, Dropdown
    interact(explore,
             T=FloatSlider(value=1.0, min=0.05, max=3.0, step=0.05, description="T (s)",
                           continuous_update=False),
             K=FloatSlider(value=1.0, min=0.25, max=3.0, step=0.25, description="K",
                           continuous_update=False),
             input=Dropdown(options=["step command r", "step disturbance d"],
                            value="step command r", description="input"))
except ImportError:
    explore()
