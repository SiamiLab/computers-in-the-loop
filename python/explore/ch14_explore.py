r"""**Tune the LQR weight.** The running plant, a zero-order hold and $1/(s(s+1))$, with state
$x = (c, \dot c)$ and state feedback $m(k) = -K x(k)$. The LQR gain minimizes
$\sum_k \big(c(k)^2 + \rho\, m(k)^2\big)$. Start from $x(0) = (1, 0)$. Lower $\rho$ (cheap control)
and the poles move toward the origin: faster, with a bigger first kick $|m(0)|$. Raise $\rho$ and
the controller gets gentle and slow. At $T = 1$ s, $\rho = 1$ gives $K = [0.7007\ \ 0.5891]$ and
poles $0.3688 \pm j0.2111$. For comparison, the gray curve places the poles at $0.5 \pm j0.5$
for every $T$.
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11})
ORANGE, BLUE, TEAL, GRAY = "#E67E22", "#1D5FA8", "#1F6E78", "#6B7785"
T_END = 10.0     # seconds shown


def _plant(T):
    p = np.exp(-T)
    return np.array([[1.0, 1 - p], [0.0, p]]), np.array([[T - 1 + p], [1 - p]])


def _lqr(Phi, Gam, rho):
    """Iterate the Riccati equation from P = 0 until it settles; return K and P."""
    Q = np.array([[1.0, 0.0], [0.0, 0.0]])
    P = np.zeros((2, 2))
    for _ in range(5000):
        s = rho + (Gam.T @ P @ Gam)[0, 0]
        P_new = Q + Phi.T @ P @ Phi - (Phi.T @ P @ Gam) @ (Gam.T @ P @ Phi) / s
        if np.max(np.abs(P_new - P)) < 1e-13:
            P = P_new
            break
        P = P_new
    K = (Gam.T @ P @ Phi) / (rho + (Gam.T @ P @ Gam)[0, 0])
    return K, P


def _ackermann(Phi, Gam, a1, a0):
    """K with det(zI - Phi + Gam K) = z^2 + a1 z + a0."""
    ctrb = np.hstack([Gam, Phi @ Gam])
    alpha = Phi @ Phi + a1 * Phi + a0 * np.eye(2)
    return np.array([[0.0, 1.0]]) @ np.linalg.solve(ctrb, alpha)


def _run(K, Phi, Gam, T, n):
    """Regulate from x(0) = (1, 0); c(t) between the samples is exact (input held)."""
    p = Phi[1, 1]
    tau = np.linspace(0, T, 21)[:-1]
    x = np.array([1.0, 0.0])
    tf, cf, ms = [], [], []
    for k in range(n):
        m = -(K @ x)[0]
        ms.append(m)
        tf.append(k * T + tau)
        cf.append(x[0] + x[1] * (1 - np.exp(-tau)) + m * (tau - 1 + np.exp(-tau)))
        x = Phi @ x + Gam[:, 0] * m
    return np.concatenate(tf), np.concatenate(cf), np.array(ms)


def explore(log10_rho=0.0, T=1.0):
    rho = 10.0 ** log10_rho
    Phi, Gam = _plant(T)
    K, P = _lqr(Phi, Gam, rho)
    poles = np.linalg.eigvals(Phi - Gam @ K)
    K_pp = _ackermann(Phi, Gam, -1.0, 0.5)             # poles 0.5 +- j0.5
    n = int(round(T_END / T))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))
    for gain, col, lab in [(K_pp, GRAY, "pole placement 0.5 $\\pm$ j0.5"),
                           (K, ORANGE, f"LQR, $\\rho$ = {rho:.3g}")]:
        tf, cf, ms = _run(gain, Phi, Gam, T, n)
        ax1.plot(tf, cf, color=col, lw=1.8 if col == ORANGE else 1.4, label=lab)
        ax1.plot(np.arange(n) * T, cf[::20], "o", color=col, ms=5 if T >= 0.3 else 3)
        ax2.step(np.arange(n + 1) * T, np.append(ms, ms[-1]), where="post", color=col,
                 lw=1.8 if col == ORANGE else 1.4, label=lab)
    for ax in (ax1, ax2):
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlim(0, T_END)
        ax.set_xlabel("t (s)")
        ax.legend(fontsize=10, loc="upper right" if ax is ax1 else "lower right")
    ax1.set_ylabel("c(t)")
    ax2.set_ylabel("m(t), held by the ZOH")
    ax1.set_title(f"regulation from x(0) = (1, 0), T = {T:g} s", fontsize=11)
    ax2.set_title("control effort", fontsize=11)
    fig.tight_layout()

    print(f"rho = {rho:.4g}, T = {T:g} s:  K = [{K[0, 0]:.4f}  {K[0, 1]:.4f}],  m(0) = {-K[0, 0]:.4f}")
    print("poles of Phi - Gamma K:", ", ".join(f"{z.real:.4f}{z.imag:+.4f}j" for z in poles),
          f"   |z| = {np.max(np.abs(poles)):.4f}")
    print(f"optimal cost from x(0) = (1, 0): x(0)' P x(0) = {P[0, 0]:.4f}")
    plt.show()


try:
    from ipywidgets import interact, FloatSlider
    interact(explore,
             log10_rho=FloatSlider(value=0.0, min=-2.0, max=2.0, step=0.1,
                                   description="log10 ρ", continuous_update=False),
             T=FloatSlider(value=1.0, min=0.2, max=2.0, step=0.1, description="T (s)",
                           continuous_update=False))
except ImportError:
    explore()
