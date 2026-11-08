r"""**One transfer function, two realizations.** $G(z) = (\beta_0 z + \beta_1)/(z^2 + \alpha_1 z + \alpha_2)$
runs here as the direct form $m_k = \beta_0 e_{k-1} + \beta_1 e_{k-2} - \alpha_1 m_{k-1} - \alpha_2 m_{k-2}$
and as the state equations $x(k+1) = Ax(k) + Be(k)$, $m_k = Cx(k)$, both from rest with a unit step.
The defaults are the chapter's example, with $m_k = 4(1 - 0.5^k)^2 \to G(1) = 4$.
Move $\alpha_1$, $\alpha_2$ and watch the poles: complex poles make the steps ring, and a pole
outside the unit circle makes them grow. The two realizations always agree.
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11})
ORANGE, BLUE, TEAL, GRAY = "#E67E22", "#1D5FA8", "#1F6E78", "#6B7785"
NK = 21     # k = 0..20


def explore(alpha1=-0.75, alpha2=0.125, beta0=1.0, beta1=0.5):
    a1, a2, b0, b1 = (round(v, 10) for v in (alpha1, alpha2, beta0, beta1))
    e = np.ones(NK)
    # direct form
    m_dir = np.zeros(NK)
    for k in range(NK):
        em1 = e[k - 1] if k >= 1 else 0.0
        em2 = e[k - 2] if k >= 2 else 0.0
        mm1 = m_dir[k - 1] if k >= 1 else 0.0
        mm2 = m_dir[k - 2] if k >= 2 else 0.0
        m_dir[k] = b0 * em1 + b1 * em2 - a1 * mm1 - a2 * mm2
    # state equations (the two-delay form as matrices)
    A = np.array([[0.0, 1.0], [-a2, -a1]])
    B = np.array([0.0, 1.0])
    C = np.array([b1, b0])
    x = np.zeros(2)
    m_ss = np.zeros(NK)
    for k in range(NK):
        m_ss[k] = C @ x
        x = A @ x + B * e[k]

    poles = np.roots([1.0, a1, a2])
    rho = np.max(np.abs(poles))
    stable = rho < 1 - 1e-12
    G1 = (b0 + b1) / (1 + a1 + a2) if abs(1 + a1 + a2) > 1e-12 else np.inf

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.0),
                                   gridspec_kw={"width_ratios": [1.9, 1]})
    k = np.arange(NK)
    ml, sl, bl = ax1.stem(k, m_dir, label="direct form")
    plt.setp(ml, color=ORANGE, markersize=6)
    plt.setp(sl, color=ORANGE, lw=2)
    plt.setp(bl, color="k", lw=0.6)
    ax1.plot(k, m_ss, "x", color=BLUE, ms=9, mew=1.6, label="state equations")
    if stable and np.isfinite(G1):
        ax1.axhline(G1, color=GRAY, ls="--", lw=1.2, label=f"G(1) = {G1:.3g}")
    ax1.set_xlabel("k")
    ax1.set_ylabel(r"$m_k$")
    ax1.set_title(rf"Unit step: $\alpha_1$ = {a1:g}, $\alpha_2$ = {a2:g}, "
                  rf"$\beta_0$ = {b0:g}, $\beta_1$ = {b1:g}", fontsize=11)
    ax1.legend(fontsize=10, loc="best")

    th = np.linspace(0, 2 * np.pi, 300)
    ax2.plot(np.cos(th), np.sin(th), color=GRAY, lw=1)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.axvline(0, color="k", lw=0.5)
    ax2.plot(poles.real, poles.imag, "x", color=ORANGE, ms=12, mew=2.5, label="poles")
    if abs(b0) > 1e-12:
        ax2.plot([-b1 / b0], [0], "o", mfc="none", color=BLUE, ms=10, mew=1.8, label="zero")
    lim = max(1.3, 1.15 * rho)
    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)
    ax2.set_aspect("equal")
    ax2.set_xlabel("Re z")
    ax2.set_ylabel("Im z")
    ax2.set_title("poles of G(z)", fontsize=11)
    ax2.legend(fontsize=10, loc="upper left")
    fig.tight_layout()

    print("poles:", ", ".join(f"{p.real:.4g}{p.imag:+.4g}j" for p in poles),
          f"  max |z| = {rho:.4g}")
    print("m_0..m_5 =", np.round(m_dir[:6], 4),
          " (both realizations agree)" if np.allclose(m_dir, m_ss, rtol=1e-9, atol=1e-9) else "")
    if stable:
        print(f"stable: m_k -> G(1) = (beta0 + beta1)/(1 + alpha1 + alpha2) = {G1:.4g}")
    else:
        print("a pole on or outside the unit circle: m_k has no final value")
    plt.show()


try:
    from ipywidgets import interact, FloatSlider
    interact(explore,
             alpha1=FloatSlider(value=-0.75, min=-2.0, max=2.0, step=0.05, description="alpha1",
                                continuous_update=False),
             alpha2=FloatSlider(value=0.125, min=-1.2, max=1.2, step=0.025, description="alpha2",
                                continuous_update=False),
             beta0=FloatSlider(value=1.0, min=-2.0, max=2.0, step=0.1, description="beta0",
                               continuous_update=False),
             beta1=FloatSlider(value=0.5, min=-2.0, max=2.0, step=0.1, description="beta1",
                               continuous_update=False))
except ImportError:
    explore()
