r"""**Saturation and windup.** A digital PI controller ($K_p = K_i = 4$, $T = 0.1$ s) drives the
plant $1/(s+1)$ through an actuator limited to $|v| \le M$. With $M = 1.2$ the plain PI keeps
integrating while the actuator is stuck: $I$ climbs to 2.77 and $c$ overshoots to 1.176. Pick an
anti-windup scheme (orange) and compare. Lower $M$ toward 1 and watch the windup get worse; raise
it and the limit stops mattering. For back-calculation, the tracking time $T_t$ sets how fast
$I$ is pulled back.
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11})
ORANGE, BLUE, TEAL, GRAY = "#E67E22", "#1D5FA8", "#1F6E78", "#6B7785"
T, KP, KI = 0.1, 4.0, 4.0
T_RUN, T_SHOW = 12.0, 8.0


def _pi_loop(mode, M, Tt):
    """mode: 'none' (no limit), 'plain', 'back-calculation', 'conditional integration'."""
    p = np.exp(-T)                                     # exact ZOH model of 1/(s+1)
    c = I = 0.0
    C, Is, V = [], [], []
    for _ in range(int(round(T_RUN / T)) + 1):
        e = 1.0 - c                                    # unit step reference
        m = KP * e + I                                 # PI output before the limit
        v = m if mode == "none" else min(M, max(-M, m))
        C.append(c)
        Is.append(I)
        V.append(v)
        if mode == "back-calculation":
            I += KI * T * e + (T / Tt) * (v - m)
        elif mode == "conditional integration":
            if abs(m) <= M or m * e < 0:
                I += KI * T * e
        else:
            I += KI * T * e
        c = p * c + (1 - p) * v
    return np.array(C), np.array(Is), np.array(V)


def _settle(C):
    """2% settling time as text; the run lasts T_RUN seconds."""
    outside = np.nonzero(np.abs(C - 1) > 0.02)[0]
    if len(outside) and outside[-1] == len(C) - 1:
        return f"> {T_RUN:g} s"
    return f"{T * (outside[-1] + 1) if len(outside) else 0.0:.1f} s"


def explore(M=1.2, scheme="back-calculation", Tt=1.0):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))
    for mode, col, lab in [("none", GRAY, "no limit"), ("plain", BLUE, "limit, plain PI"),
                           (scheme, ORANGE, f"limit, {scheme}")]:
        C, I, V = _pi_loop(mode, M, Tt)
        t = T * np.arange(len(C))
        keep = t <= T_SHOW + 1e-9
        ls = "--" if mode == "none" else "-"
        ax1.plot(t[keep], C[keep], ls, color=col, lw=1.8, label=lab)
        ax2.plot(t[keep], I[keep], ls, color=col, lw=1.8, label=lab)
        stuck = int(np.sum(np.abs(V) >= M - 1e-12)) if mode != "none" else 0
        print(f"{lab:38s}: peak c = {C.max():.3f}, largest I = {I.max():.2f}, "
              f"2% settling = {_settle(C)}" + (f", samples at the limit = {stuck}" if mode != "none" else ""))
    ax1.axhline(1, color="k", lw=0.6)
    ax2.axhline(M, color="k", lw=0.8, ls=":", label=f"limit M = {M:g}")
    for ax in (ax1, ax2):
        ax.set_xlim(0, T_SHOW)
        ax.set_xlabel("t (s)")
        ax.legend(fontsize=10, loc="lower right")
    ax1.set_ylabel("c(kT)")
    ax2.set_ylabel("integrator state I(k)")
    extra = f", $T_t$ = {Tt:g} s" if scheme == "back-calculation" else ""
    ax1.set_title(f"output, |v| $\\leq$ {M:g}{extra}", fontsize=11)
    ax2.set_title("integrator state", fontsize=11)
    fig.tight_layout()
    plt.show()


try:
    from ipywidgets import interact, FloatSlider, Dropdown
    interact(explore,
             M=FloatSlider(value=1.2, min=1.05, max=4.0, step=0.05, description="limit M",
                           continuous_update=False),
             scheme=Dropdown(options=["back-calculation", "conditional integration"],
                             value="back-calculation", description="anti-windup"),
             Tt=FloatSlider(value=1.0, min=0.1, max=5.0, step=0.1, description="T_t (s)",
                            continuous_update=False))
except ImportError:
    explore()
