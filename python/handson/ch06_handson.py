r"""**Hands-on, Chapter 6.** These cells go with the Hands-on exercises at the end of
Chapter 6. Do each exercise on paper first and write down your prediction; then use the
matching section below to check it. Each section has sliders: change one thing at a time.
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11})
ORANGE, BLUE, TEAL, GRAY = "#E67E22", "#1D5FA8", "#1F6E78", "#6B7785"


def signed_alias(f, fs):
    """The frequency in [-fs/2, fs/2) that has the same samples as f (sign included)."""
    return f - fs * np.round(f / fs)


# ---------------------------------------------------------------- H1
def h1_backward_vibration(f_vib=110.0, fs=120.0):
    """H1: a sine at f_vib sampled at fs looks like a sine at the signed alias frequency."""
    T = 1 / fs
    t = np.linspace(0, 0.25, 6000)
    k = np.arange(int(0.25 * fs) + 1)
    fa = signed_alias(f_vib, fs)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(t, np.sin(2 * np.pi * f_vib * t), color=GRAY, lw=0.6, label=f"v(t), {f_vib:g} Hz")
    ax.plot(t, np.sin(2 * np.pi * fa * t), "--", color=BLUE, lw=1.6,
            label=f"sin(2 pi ({fa:+.1f}) t): same samples")
    ax.plot(k * T, np.sin(2 * np.pi * f_vib * k * T), "o", color=ORANGE, ms=6, label="samples")
    ax.set_xlabel("t (s)")
    ax.set_ylim(-1.6, 1.6)
    sign = "inverted" if fa < 0 else "upright"
    ax.set_title(f"fs = {fs:g} Hz: the loop sees {abs(fa):.1f} Hz, {sign}", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    plt.show()


# ---------------------------------------------------------------- H2
def h2_double_pole(K=4.0, T=10.0):
    """H2: c(t) = K t exp(-a t) with exp(-aT) = 1/2; correct samples vs the simple-pole slip."""
    a = np.log(2) / 10
    t = np.linspace(0, 80, 800)
    k = np.arange(int(80 // T) + 1)
    q = np.exp(-a * T)
    right = K * T * k * q ** k
    wrong = K * q ** k
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(t, K * t * np.exp(-a * t), color=GRAY, lw=1.4, label="c(t)")
    ax.plot(k * T, right, "o", color=ORANGE, ms=7, label="samples, double-pole residue")
    ax.plot(k * T, wrong, "s--", color=TEAL, ms=5, lw=1, label="simple-pole slip")
    ax.axhline(K / (a * np.e), color=BLUE, ls=":", lw=1, label=f"true peak {K / (a * np.e):.2f}")
    ax.set_xlabel("t (min)")
    ax.set_ylabel("mU/L")
    ax.set_title(f"T = {T:g} min: largest sample {right.max():.2f}", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    plt.show()


# ---------------------------------------------------------------- H3
def h3_ripple_offset(fs=40.0, phi_deg=30.0):
    """H3: v(t) = A sin(100 pi t + phi) sampled at fs; spot the rates with constant samples."""
    A = 0.4
    phi = np.radians(phi_deg)
    T = 1 / fs
    t = np.linspace(0, 0.4, 8000)
    k = np.arange(int(0.4 * fs) + 1)
    vk = A * np.sin(100 * np.pi * k * T + phi)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(t, A * np.sin(100 * np.pi * t + phi), color=GRAY, lw=0.6, label="ripple v(t)")
    ax.plot(k * T, vk, "o-", color=ORANGE, ms=6, lw=1, label="samples")
    ax.axhline(A * np.sin(phi), color=BLUE, ls="--", lw=1, label=f"A sin(phi) = {A * np.sin(phi):.3f}")
    ax.set_ylim(-0.6, 0.6)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("deg C")
    ax.set_title(f"fs = {fs:g} Hz: samples span {vk.min():+.3f} to {vk.max():+.3f} deg C",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    plt.show()


# ---------------------------------------------------------------- H4
def h4_fold_band(fs=60.0):
    """H4: fold the noise band 170-190 Hz for a given fs and test it against the 5 Hz limit."""
    f = np.linspace(170, 190, 2001)
    folded = np.abs(signed_alias(f, fs))
    ok = folded.min() >= 5 - 1e-9
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.8), gridspec_kw={"width_ratios": [2, 1]})
    ax1.plot(f, folded, color=ORANGE, lw=2)
    ax1.axhspan(0, 5, color=GRAY, alpha=0.25, label="loop band, below 5 Hz")
    ax1.set_xlabel("noise frequency (Hz)")
    ax1.set_ylabel("frequency seen (Hz)")
    ax1.set_ylim(0, 52)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_title(f"fs = {fs:g} Hz: folded band {folded.min():.1f} to {folded.max():.1f} Hz, "
                  f"{'PASS' if ok else 'FAIL'}", fontsize=11)
    sweep = np.linspace(40, 100, 1201)
    worst = [np.abs(signed_alias(f, s)).min() for s in sweep]
    ax2.plot(sweep, worst, color=BLUE, lw=1.4)
    ax2.axhline(5, color=TEAL, ls="--", lw=1)
    ax2.axvline(fs, color=ORANGE, lw=1)
    ax2.set_xlabel("fs (Hz)")
    ax2.set_title("lowest folded frequency", fontsize=11)
    plt.show()


try:
    from ipywidgets import interact, FloatSlider
    interact(h1_backward_vibration,
             f_vib=FloatSlider(value=110, min=100, max=140, step=1, description="f vib (Hz)",
                               continuous_update=False),
             fs=FloatSlider(value=120, min=60, max=240, step=5, description="fs (Hz)",
                            continuous_update=False))
    interact(h2_double_pole, K=FloatSlider(value=4, min=1, max=8, step=0.5, description="K",
                                           continuous_update=False),
             T=FloatSlider(value=10, min=2, max=20, step=1, description="T (min)",
                           continuous_update=False))
    interact(h3_ripple_offset,
             fs=FloatSlider(value=40, min=10, max=120, step=0.01, description="fs (Hz)",
                            continuous_update=False),
             phi_deg=FloatSlider(value=30, min=0, max=360, step=5, description="phi (deg)",
                                 continuous_update=False))
    interact(h4_fold_band, fs=FloatSlider(value=60, min=40, max=100, step=0.25, description="fs (Hz)",
                                          continuous_update=False))
except ImportError:
    h1_backward_vibration()
    h2_double_pole()
    h3_ripple_offset()
    h4_fold_band()
