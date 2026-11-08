r"""**Aliasing.** Sample $\cos(2\pi f_0 t)$ at $f_s$ samples per second. After sampling, a
frequency $f_0$ shows up at $|f_0 - n f_s|$, with the integer $n$ that puts it in
$[0, f_s/2]$. Start at the chapter's example, $f_0 = 7$ Hz and $f_s = 10$ Hz: the samples
are exactly those of a 3 Hz cosine. Raise $f_s$ past $2f_0$ and the alias disappears; set
$f_0 = f_s$ and the samples look like a constant.
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11})
ORANGE, BLUE, TEAL, GRAY = "#E67E22", "#1D5FA8", "#1F6E78", "#6B7785"


def explore(f0=7.0, fs=10.0):
    f0, fs = round(f0, 10), round(fs, 10)
    T = 1 / fs
    n = np.round(f0 / fs)
    fa = abs(f0 - n * fs)                     # the frequency the samples show
    window = max(0.5, 5 * T)
    t = np.linspace(0, window, 3000)
    kk = np.arange(int(np.floor(window / T + 1e-9)) + 1)
    samples = np.cos(2 * np.pi * f0 * kk * T)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.0),
                                   gridspec_kw={"width_ratios": [1.6, 1]})
    ax1.plot(t, np.cos(2 * np.pi * f0 * t), color=ORANGE, lw=1.6, label=f"{f0:g} Hz signal")
    if abs(fa - f0) > 1e-9:
        ax1.plot(t, np.cos(2 * np.pi * fa * t), "--", color=TEAL, lw=1.8,
                 label=f"{fa:g} Hz alias")
    ax1.plot(kk * T, samples, "o", color=BLUE, ms=6, label=f"samples, $f_s$ = {fs:g} Hz")
    ax1.set_xlim(0, window)
    ax1.set_ylim(-1.3, 1.65)
    ax1.set_xlabel("t (s)")
    ax1.set_ylabel(r"$\cos(2\pi f_0 t)$")
    ax1.set_title(f"$f_0$ = {f0:g} Hz sampled at $f_s$ = {fs:g} Hz (T = {T:.3g} s)", fontsize=11)
    ax1.legend(fontsize=10, loc="upper right", ncol=3)

    # spectrum of the sampled signal: lines at +-f0 + n fs
    fmax = max(1.5 * fs, 1.2 * f0)
    ax2.axvspan(-fs / 2, fs / 2, color=TEAL, alpha=0.12, lw=0)
    for m in range(-int(np.ceil(fmax / fs)) - 2, int(np.ceil(fmax / fs)) + 3):
        for sgn in (1, -1):
            f = sgn * f0 + m * fs
            if abs(f) <= fmax:
                is_orig = (m == 0)
                ax2.plot([f, f], [0, 1], color=ORANGE if is_orig else GRAY,
                         lw=2.6 if is_orig else 1.4, zorder=3 if is_orig else 2)
    if abs(fa - f0) > 1e-9:
        for f in (fa, -fa):
            ax2.plot([f], [1.0], "v", color=TEAL, ms=9, zorder=4)
    ax2.axvline(fs / 2, color=TEAL, lw=1, ls=":")
    ax2.axvline(-fs / 2, color=TEAL, lw=1, ls=":")
    ax2.set_xlim(-fmax, fmax)
    ax2.set_ylim(0, 1.25)
    ax2.set_yticks([])
    ax2.set_xlabel("f (Hz)")
    ax2.set_title(r"spectrum lines: $\pm f_0$ (orange), copies every $f_s$", fontsize=10)
    fig.tight_layout()

    print(f"f_s = {fs:g} Hz, Nyquist frequency f_s/2 = {fs / 2:g} Hz")
    if f0 < fs / 2 - 1e-9:
        print(f"f0 = {f0:g} Hz < f_s/2: no aliasing, the samples show {f0:g} Hz")
    elif abs(f0 - fs / 2) < 1e-9:
        print(f"f0 = f_s/2 exactly: two samples per period, on the edge (a sine would give all zeros)")
    else:
        print(f"f0 = {f0:g} Hz > f_s/2: aliased, it shows up at |{f0:g} - {int(n)}*{fs:g}| = {fa:g} Hz"
              + (" (a constant)" if fa < 1e-9 else ""))
    print(f"largest gap between the samples and the {fa:g} Hz cosine: "
          f"{np.max(np.abs(samples - np.cos(2 * np.pi * fa * kk * T))):.1e}")
    plt.show()


try:
    from ipywidgets import interact, FloatSlider
    interact(explore,
             f0=FloatSlider(value=7.0, min=0.0, max=25.0, step=0.5, description="f0 (Hz)",
                            continuous_update=False),
             fs=FloatSlider(value=10.0, min=1.0, max=30.0, step=0.5, description="fs (Hz)",
                            continuous_update=False))
except ImportError:
    explore()
