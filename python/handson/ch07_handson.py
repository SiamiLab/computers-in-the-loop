r"""**Hands-on, Chapter 7.** These cells go with the Hands-on exercises at the end of
Chapter 7. Do each exercise on paper first and write down your prediction; then use the
matching section below to check it. Each section has sliders: change one thing at a time.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

plt.rcParams.update({"font.size": 11})
ORANGE, BLUE, TEAL, GRAY = "#E67E22", "#1D5FA8", "#1F6E78", "#6B7785"


def staircase(fs, f, t, phase=0.0):
    """Zero-order hold of the samples of sin(2 pi f t + phase) taken at fs."""
    return np.sin(2 * np.pi * f * np.floor(t * fs) / fs + phase)


def line_amplitudes(x, dt, freqs):
    """Amplitude of the sinusoidal component of x at each frequency (whole periods assumed)."""
    t = np.arange(len(x)) * dt
    return np.array([2 * abs(np.mean(x * np.exp(-2j * np.pi * fr * t))) for fr in freqs])


# ---------------------------------------------------------------- H1
def h1_staircase_spectrum(f=4.0, fs=100.0):
    """H1: a held sine; its fundamental and the images at n fs +- f."""
    dt = 1 / (fs * 200)
    t = np.arange(0, 1.0, dt)                     # 1 s: whole periods of f and of fs
    ub = staircase(fs, f, t)
    lines = [f] + [n * fs + s * f for n in (1, 2) for s in (-1, 1)]
    amps = line_amplitudes(ub, dt, lines)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.8), gridspec_kw={"width_ratios": [3, 2]})
    show = t < 2 / f
    ax1.plot(t[show], np.sin(2 * np.pi * f * t[show]), color=GRAY, lw=1.2, label="sine")
    ax1.plot(t[show], ub[show], color=ORANGE, lw=1.4, label="staircase")
    ax1.set_xlabel("t (s)")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_title(f"f = {f:g} Hz, fs = {fs:g} Hz", fontsize=11)
    ax2.stem(lines, amps, linefmt=BLUE, markerfmt="o", basefmt=" ")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-3, 2)
    ax2.set_xlabel("frequency (Hz)")
    ax2.set_title(f"image/fundamental {amps[1] / amps[0]:.4f}; f/(fs-f) = {f / (fs - f):.4f}",
                  fontsize=10)
    plt.show()


# ---------------------------------------------------------------- H2
def h2_pitch_lag(fs=20.0, wc=3.0, pm=50.0):
    """H2: sampler + ZOH seen as a gain and a phase at the crossover frequency."""
    T = 1 / fs
    w = np.linspace(0.01, 2 * np.pi * fs, 2000)
    x = w * T / 2
    gain = np.abs(np.sin(x) / x)                  # sampler (1/T) times hold (T sinc)
    lag = np.degrees(x)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.6))
    ax1.plot(w, gain, color=BLUE, lw=1.6)
    ax1.axvline(wc, color=ORANGE, ls="--", lw=1)
    ax1.set_xlabel("omega (rad/s)")
    ax1.set_title("gain of sampler + ZOH", fontsize=11)
    ax2.plot(w, lag, color=TEAL, lw=1.6)
    ax2.axvline(wc, color=ORANGE, ls="--", lw=1)
    ax2.set_xlabel("omega (rad/s)")
    ax2.set_title(f"lag at wc: {np.degrees(wc * T / 2):.2f} deg, PM about "
                  f"{pm - np.degrees(wc * T / 2):.1f} deg", fontsize=11)
    plt.show()


# ---------------------------------------------------------------- H3
def h3_flexible_arm(f=8.0, fs=50.0):
    """H3: held sine into a lightly damped 45 Hz mode; largest fast component of the tip motion."""
    wr, z = 2 * np.pi * 45, 0.01
    arm = signal.lti([wr ** 2], [1, 2 * z * wr, wr ** 2])
    dt = 1 / (fs * 40)
    t = np.arange(0, 8.0, dt)
    ub = staircase(fs, f, t)
    _, y, _ = signal.lsim(arm, ub, t, interp=False)
    keep = t >= 6.0                              # last 2 s, after the mode has settled
    tk, yk = t[keep], y[keep]
    spec = 2 * np.abs(np.fft.rfft(yk)) / len(yk)
    freqs = np.fft.rfftfreq(len(yk), dt)
    fast = freqs > 20
    i = np.argmax(spec * fast)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    show = tk < 6.4
    ax.plot(tk[show], ub[keep][show], color=GRAY, lw=1, label="held command")
    ax.plot(tk[show], yk[show], color=ORANGE, lw=1.2, label="tip position")
    ax.set_xlabel("t (s)")
    ax.set_ylim(-8, 8)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"f = {f:g} Hz, fs = {fs:g} Hz: largest fast component {spec[i]:.2f} A "
                 f"at {freqs[i]:.1f} Hz", fontsize=11)
    plt.show()


# ---------------------------------------------------------------- H4
def h4_fractional_hold(beta=0.45, sigma=1.0):
    """H4: fractional-order hold; low-frequency delay, rms of held noise, step overshoot."""
    rng = np.random.default_rng(1)
    n, m = 4000, 50                               # samples, points per period
    eps = sigma * rng.standard_normal(n)
    x = np.arange(m) / m
    held = [(1 + beta * x) * eps[k] - beta * x * eps[k - 1] for k in range(1, n)]
    rms = np.sqrt(np.mean(np.concatenate(held) ** 2))
    u = np.array([0.0, 0, 1, 1, 1, 1])
    xe = np.linspace(0, 1, m)                     # up to the instant just before the next sample
    step = np.concatenate([u[k] + beta * (u[k] - u[k - 1]) * xe for k in range(1, len(u))])
    ts = np.arange(len(step)) / m
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(ts, step, color=ORANGE, lw=1.6, label="held step")
    ax.axhline(1.5, color=TEAL, ls="--", lw=1, label="50% overshoot limit")
    ax.set_xlabel("t / T")
    ax.set_ylim(-0.1, 2.2)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"beta = {beta:.2f}: delay {(1 - beta) / 2:.3f} T, rms {rms / sigma:.3f} sigma "
                 f"(formula {np.sqrt(1 + beta + 2 * beta ** 2 / 3):.3f}), overshoot "
                 f"{100 * (step.max() - 1):.0f}%", fontsize=10)
    plt.show()


try:
    from ipywidgets import interact, FloatSlider
    interact(h1_staircase_spectrum,
             f=FloatSlider(value=4, min=1, max=25, step=1, description="f (Hz)",
                           continuous_update=False),
             fs=FloatSlider(value=100, min=50, max=400, step=50, description="fs (Hz)",
                            continuous_update=False))
    interact(h2_pitch_lag, fs=FloatSlider(value=20, min=2, max=50, step=0.5, description="fs (Hz)",
                                          continuous_update=False),
             wc=FloatSlider(value=3, min=0.5, max=10, step=0.5, description="wc (rad/s)",
                            continuous_update=False),
             pm=FloatSlider(value=50, min=20, max=80, step=5, description="PM (deg)",
                            continuous_update=False))
    interact(h3_flexible_arm, f=FloatSlider(value=8, min=1, max=20, step=0.5, description="f (Hz)",
                                            continuous_update=False),
             fs=FloatSlider(value=50, min=30, max=80, step=5, description="fs (Hz)",
                            continuous_update=False))
    interact(h4_fractional_hold, beta=FloatSlider(value=0.45, min=0, max=1, step=0.05,
                                                  description="beta", continuous_update=False),
             sigma=FloatSlider(value=1, min=0.5, max=2, step=0.5, description="sigma",
                               continuous_update=False))
except ImportError:
    h1_staircase_spectrum()
    h2_pitch_lag()
    h3_flexible_arm()
    h4_fractional_hold()
