# MATLAB companion scripts

*Computers in the Loop: Digital Control from Sampling to Networked Agents*, by Milad Siami.

One MATLAB script per chapter. Each script reproduces the main worked examples of its
chapter: it prints the key numbers, labeled, and saves its figures as PNG files in
`matlab/figures/` (the folder is created if needed). After every printed result the script
has a comment `% expected: ...` with the value you should see, so you can check your run
and then change the numbers to try other cases.

## Requirements

- MATLAB R2016b or later (the scripts use local functions at the end of a script file).
- Base MATLAB only: `roots`, `poly`, `conv`, `filter`, `residue`, `expm`, `eig`, `fzero`,
  `integral` and the plotting functions.
- The Control System Toolbox is optional. Where it gives a shorter route (`tf`, `c2d`,
  `step`, `rlocus`, `margin`, `nyquist`, `acker`, `dlqr`, `dlyap`), the toolbox one-liner is
  shown in a comment next to the base-MATLAB code, so you can compare the two.

A reminder on `filter`: its coefficient vectors are in powers of z^-1. For example
z/(z^2 - 3z + 2) = z^-1/(1 - 3z^-1 + 2z^-2) is `filter([0 1], [1 -3 2], x)`.

## How to run

Open MATLAB in the `matlab/` folder and type the script name, for example

```
ch01_laser_loop
```

Each script starts with `clear` and `close all`, so run them one at a time.

## What each script reproduces

| Script | Chapter | What it reproduces |
|---|---|---|
| `ch01_laser_loop.m` | 1, Introduction to Digital Control | The laser example: the analog integral design (a = 2, K = 1, 2% settling 5.83 s); the same controller on a computer for T = 0.1 to 2.5 s (step and disturbance peaks, size of the closed-loop roots, the limit T = 4/a = 2 s); the phase margin from the delay estimate and from the sampled loop; the 12-bit quantization step. Figure `ch01_laser_step.png`. |
| `ch02_mason.m` | 2, Block Diagrams and Mason's Gain Formula | The robot-arm position loop (J, B, K, a, the closed loop 25/(s^2 + 6s + 25), the effect of the motor inductance); Mason's gain formula (paths, loops, Delta, cofactors) checked against the node equations on Example 2 (5/8), Example 3 (three interlaced loops, against the block-diagram reduction) and the rate-feedback loop. Figure `ch02_arm_step.png`. |
| `ch03_z_transform.m` | 3, Discrete-Time Systems and the z-Transform | The satellite through a sample-and-hold (difference equation, theta(k) = k^2/2, pulse transfer function); the z-transform series against closed forms inside and outside the region of convergence; the shift and scaling rules; the position loop with K = 1 (stable) and K = 3 (unstable), where the final value formula still returns 0. Figure `ch03_position_error.png`. |
| `ch04_inverse_z.m` | 4, Difference Equations and the Inverse z-Transform | One convolution two ways (2 - 0.5^k); a difference equation with a step input ((-1)^k); z/((z-1)(z-2)) inverted by partial fractions, long division and residues (2^k - 1); the repeated-pole example; 1/(z - 0.5); Parseval (4/3); the practice exercise. Figure `ch04_convolution_parseval.png`. |
| `ch05_realizations.m` | 5, Realizations, State Variables, and the Sample-and-Hold | G(z) = (z + 0.5)/(z^2 - 0.75z + 0.125) run as the direct form, the two-delay form and the state equations (all give 4(1 - 0.5^k)^2); C(zI - A)^-1 B = G(z); the sampled-signal table on 1/((s+1)(s+2)) with T = ln 2; a ramp through the hold and the hold-error bound. Figure `ch05_realizations.png`. |
| `ch06_starred_aliasing.m` | 6, The Starred Transform, Aliasing, and the Zero-Order Hold | Starred transforms by series and by residues (Examples 1 and 2); Example 3 with a delay, where residues on E(s) as it stands give wrong samples and taking the delay out first fixes them; aliasing (7 Hz and 3 Hz at 10 Hz), hidden oscillations; ZOH gain and phase at 1 Hz and 5 Hz for T = 0.1 s. Figure `ch06_aliasing_zoh.png`. |
| `ch07_holds.m` | 7, Data Reconstruction and Data Holds | Gain and phase of the zero-order, fractional-order (beta = 0.5) and first-order holds at wT = 0.5 and at ws/2; the frequency where the ZOH and FOH lags are equal; the FOH peak gain; the images of a held sinusoid; the low-frequency delay (1 - beta)T/2. Figure `ch07_holds.png`. |
| `ch08_open_loop.m` | 8, Open-Loop Sampled-Data Systems | E(z) of 1/((s+1)(s+2)) at T = 0.5 s; the pulse transfer function of ZOH + 1/(s+1) by brute force and by exact discretization, its step samples and DC gain; where the samplers sit (configurations (a) and (b)); the digital-filter example; how large T may be for 25/(s^2 + 2s + 25) (T = 1 s and 0.1 s, true peak against the samples). Figure `ch08_how_large_T.png`. |
| `ch09_modified_z.m` | 9, The Modified z-Transform and Closed-Loop Systems | Examples A1, A2, A3 of the modified z-transform (A3: G(z,0.6) = (0.4512z + 0.1809)/(z(z - 0.3679)) and its step samples); two samplers that are not synchronized (c(T) = 0.000398); closed loops B1 and B3 checked by simulating the sampled loop with the plant integrated exactly. Figure `ch09_modified_z.png`. |
| `ch10_time_response.m` | 10, Closed-Loop Time Response and Steady-State Accuracy | Example 1 (first-order plant, T = 0.1 s, pole 0.4562, tau = 0.127 s); Example 2 (1/(s(s+1)), T = 1 s: zeta = 0.25, wn = 0.92, tau = 4.36 s, peak 1.40 at the samples and 1.45 between them); the effect of T and the limit T = 3.92 s; Example B (digital integrator); the laser loop gain limits (54.7 at T = 0.1 s, 6.78 at T = 1 s); the practice exercise with K = 2. Figure `ch10_time_response.png`. |
| `ch11_stability.m` | 11, Stability of Sampled-Data Systems | The sampled integrator (stable for K < 2/T); Routh's array with the epsilon rule and the auxiliary polynomial on the chapter's examples (including the gain range 0 < K < 204/25); the bilinear substitution plus Routh, and Jury's test, on the running loop (0 < K < 2.39), the loop with one sample of delay (0 < K < 0.805), the laser loop (k < 54.7) and the practice exercise (0 < K < 6), each cross-checked with the roots. Figure `ch11_stability.png`. |
| `ch12_root_locus_frequency.m` | 12, Root Locus and Frequency Response of Sampled-Data Loops | Root locus of the running loop (circle, breakaway 0.648, crossing of the unit circle at K = 2.39); the gain for zeta = 0.5; the effect of T on the critical gain; the w-plane Bode plot, gain margin 2.39 (7.58 dB) and phase margin 30.4 degrees at K = 1; the Nyquist plot and Z = N + P; the locus with one sample of delay. Figures `ch12_root_locus.png`, `ch12_bode_w.png`, `ch12_nyquist.png`, `ch12_root_locus_delay.png`. |
| `ch13_controller_design.m` | 13, Designing Digital Controllers | One lead compensator converted six ways (forward and backward Euler, Tustin, prewarped Tustin, matched pole-zero, ZOH equivalent) and the closed-loop responses; a lead designed directly in z for the running plant; deadbeat designs with and without ripple between the samples; a digital PID with a derivative filter. Figures `ch13_emulation.png`, `ch13_direct_lead.png`, `ch13_deadbeat.png`, `ch13_pid.png`. |
| `ch14_state_space.m` | 14, State-Space Design: Pole Placement, Observers, and Optimal Control | Phi and Gamma of the running plant; controllability and observability; pole placement by Ackermann's formula; deadbeat feedback; reference gain and integral action; prediction and current observers; the controller-observer as D(z); discrete LQR by Riccati iteration. Figures `ch14_pole_placement.png`, `ch14_integral_action.png`, `ch14_observer.png`, `ch14_lqr.png`. |
| `ch15_implementation.m` | 15, Implementing Digital Controllers | Choosing T; the anti-aliasing prefilter and its phase; one sample of computation delay; quantization noise gains; coefficient rounding in direct and cascade forms; deadband and limit cycles; integrator windup and two anti-windup schemes. Figures `ch15_pm_vs_T.png`, `ch15_delay_steps.png`, `ch15_noise_gain.png`, `ch15_coef_rounding.png`, `ch15_limit_cycles.png`, `ch15_windup.png`. |
| `ch16_networks_sampling.m` | 16, How Fast Must Networked Agents Sample? | Agreement by hand (two agents, three agents in a line); the largest stable update period of one agent; Laplacian eigenvalues of path, ring, star and complete graphs and the largest stable T of each (formula, bisection, exact model, platoon); fast and slow agents simulated; links added one at a time (8 agents); the per-mode noise variance, checked with the discrete Lyapunov equation, and the sparse-dense crossing. Figures `ch16_agreement.png`, `ch16_roots.png`, `ch16_tmax_vs_lmax.png`, `ch16_fast_slow.png`, `ch16_density.png`, `ch16_crossing.png`. |

## Notes

- Where MATLAB returns roots, residues or eigenvalues in a different order from the one in
  the notes, the `% expected:` comment says so; the values are the same.
- The round-off noise simulation of Chapter 15 drives the filter with random inputs
  (`rng(1)`), so its simulated variances are close to, but not exactly, the values in the
  `% expected:` comments; they stay close to the formula.
- Tiny numbers that are zero up to round-off (for example 2e-15) can differ in the last
  digits from one computer to another.

The appendix of tables and formulas (Appendix B) collects the transform pairs, the
zero-order-hold equivalents and the stability tests that these scripts use.
