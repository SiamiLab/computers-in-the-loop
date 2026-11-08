%% Computers in the Loop, Chapter 15: Implementing Digital Controllers
% Choosing T, the anti-aliasing prefilter, one sample of computation delay,
% quantization noise, coefficient rounding in direct and cascade forms,
% deadband and limit cycles, and integrator windup with two anti-windup schemes.
% Milad Siami
%
% Figures (in matlab/figures/):
%   ch15_pm_vs_T.png        phase margin against T, with and without the anti-aliasing filter
%   ch15_delay_steps.png    step responses with and without one sample of computation delay
%   ch15_noise_gain.png     round-off noise in a first-order filter: formula against simulation
%   ch15_coef_rounding.png  poles of a 4th-order filter after rounding: direct form vs cascade
%   ch15_limit_cycles.png   deadband and limit cycle of a rounded first-order recursion
%   ch15_windup.png         PI control with a saturating actuator: windup and anti-windup
%
% Change the parameters at the top of each section to experiment.
% Run this file from the matlab/ folder.

clear; close all;
here = fileparts(mfilename('fullpath'));
figdir = fullfile(here, 'figures');
if ~exist(figdir, 'dir'), mkdir(figdir); end

%% 1. Sampling period and anti-aliasing prefilter (plant 1/(s(s+1)), K = 1)
ATTENUATION_AT_NYQUIST = 0.1;         % prefilter gain at ws/2 (0.1 = -20 dB)
fprintf('\n1. Sampling period and prefilter (plant 1/(s(s+1)), K = 1)\n');
wc = sqrt((sqrt(5) - 1)/2);           % analog crossover
pmAnalog = 90 - atand(wc);
fprintf('  analog design: w_c = %.3f rad/s, PM = %.1f deg\n', wc, pmAnalog);
% expected: w_c = 0.786 rad/s, PM = 51.8 deg
% Butterworth prefilter cutoff from |H_f(j wN)| = 1/sqrt(1 + (wN/wf)^4) = attenuation
ratio = (1/ATTENUATION_AT_NYQUIST^2 - 1)^0.25;
Ts = 0.02*(1:20);                     % T = 0.02, 0.04, ..., 0.40 s
pmHold = zeros(size(Ts));  pmBoth = zeros(size(Ts));
for i = 1:numel(Ts)
    wf = (pi/Ts(i))/ratio;
    pmHold(i) = phase_margin(Ts(i), 1, [], 0);
    pmBoth(i) = phase_margin(Ts(i), 1, wf, 0);
end
for T = [0.05 0.1 0.2]
    wf = (pi/T)/ratio;
    fprintf('  T = %4.2f s: w_f = %6.2f rad/s, PM hold only = %.1f deg, hold + prefilter = %.1f deg\n', ...
            T, wf, phase_margin(T, 1, [], 0), phase_margin(T, 1, wf, 0));
end
% expected:
%   T = 0.05 s: w_f =  19.92 rad/s, PM hold only = 50.7 deg, hold + prefilter = 47.5 deg
%   T = 0.10 s: w_f =   9.96 rad/s, PM hold only = 49.6 deg, hold + prefilter = 43.2 deg
%   T = 0.20 s: w_f =   4.98 rad/s, PM hold only = 47.3 deg, hold + prefilter = 34.5 deg
kpf = sqrt(2)*ratio/pi;               % the prefilter acts like a delay of kpf*T
fig = figure('Color', 'w', 'Position', [100 100 640 380]);
hold on; box on; grid on;
h1 = plot(Ts, pmHold, 'o-');
h2 = plot(Ts, pmBoth, 's-');
h3 = plot(Ts, pmAnalog - wc*Ts/2*180/pi, '--');
h4 = plot(Ts, pmAnalog - (0.5 + kpf)*wc*Ts*180/pi, '--');
plot([Ts(1) Ts(end)], [pmAnalog pmAnalog], ':', 'Color', [0.5 0.5 0.5]);
xlabel('T (s)'); ylabel('phase margin (deg)');
legend([h1 h2 h3 h4], {'hold only (exact)', 'hold and prefilter (exact)', ...
       'estimate: hold = delay T/2', sprintf('estimate: hold + prefilter = delay %.2f T', 0.5 + kpf)}, ...
       'FontSize', 8, 'Location', 'southwest');
print(fig, fullfile(figdir, 'ch15_pm_vs_T.png'), '-dpng', '-r150');
fprintf('  saved figures/ch15_pm_vs_T.png\n');

%% 2. Computation delay in the running example (ZOH + 1/(s(s+1)), T = 1 s, D(z) = K)
fprintf('\n2. Computation delay (running example, T = 1 s)\n');
for frac = [1e-9, 0.1, 0.2, 0.5, (1 - 1e-9)]
    fprintf('  output %4.2f T after the sample: stable for 0 < K < %.2f\n', frac, largest_stable_gain(frac));
end
% expected: 2.39, 1.94, 1.65, 1.17, 0.80
cases = [0.5 0; 0.5 1; 0.3 1];        % [K, delay in samples]
for i = 1:3
    K = cases(i, 1);  d = cases(i, 2);
    pm = phase_margin(1, K, [], d);
    c = step_response(K, d, 80);
    fprintf('  K = %g, delay = %d: PM = %.1f deg, overshoot = %.1f %%\n', K, d, pm, 100*(max(c) - 1));
end
% expected:
%   K = 0.5, delay = 0: PM = 52.8 deg, overshoot = 14.6 %
%   K = 0.5, delay = 1: PM = 26.9 deg, overshoot = 52.7 %
%   K = 0.3, delay = 1: PM = 49.3 deg, overshoot = 19.7 %
fig = figure('Color', 'w', 'Position', [100 100 640 360]);
hold on; box on; grid on;
marks = {'o-', 's-', '^-'};
h = gobjects(0);  labels = {};
for i = 1:3
    K = cases(i, 1);  d = cases(i, 2);
    h(end+1) = plot(0:39, step_response(K, d, 40), marks{i}, 'MarkerSize', 3);
    if d, txt = 'one sample of delay'; else, txt = 'no delay'; end
    labels{end+1} = sprintf('K = %g, %s', K, txt);
end
plot([0 39], [1 1], ':', 'Color', [0.5 0.5 0.5]);
xlabel('k'); ylabel('c(kT)');
legend(h, labels, 'FontSize', 8, 'Location', 'southeast');
print(fig, fullfile(figdir, 'ch15_delay_steps.png'), '-dpng', '-r150');
fprintf('  saved figures/ch15_delay_steps.png\n');

%% 3. Quantization noise through m(k) = Q[a m(k-1) + (1-a) e(k)] (units of q)
% MATLAB's round() rounds to the nearest integer, ties away from zero.
fprintf('\n3. Round-off noise in m(k) = Q[a m(k-1) + (1-a) e(k)]   (units of q)\n');
rng(1);                               % repeatable random input
poles = [0.5 0.8 0.9 0.92 0.95 0.98];
predicted = zeros(size(poles));  simulated = zeros(size(poles));
for i = 1:numel(poles)
    a = poles(i);
    e = -500 + 1000*rand(1, 60000);   % a busy input, so the error looks random
    mq = 0;  mi = 0;
    err = zeros(1, 60000);
    for k = 1:60000
        mq = round(a*mq + (1 - a)*e(k));   % what the processor computes
        mi = a*mi + (1 - a)*e(k);          % infinite precision
        err(k) = mq - mi;
    end
    predicted(i) = 1/(1 - a^2)/12;      % noise gain times q^2/12
    simulated(i) = var(err(2001:end), 1);
    fprintf('  a = %4.2f: variance formula %7.3f, simulated %7.3f\n', a, predicted(i), simulated(i));
end
% expected (the simulated values depend on the random input; they come out close to):
%   a = 0.50: formula 0.111, simulated 0.111
%   a = 0.80: formula 0.231, simulated 0.227
%   a = 0.90: formula 0.439, simulated 0.435
%   a = 0.92: formula 0.543, simulated 0.557
%   a = 0.95: formula 0.855, simulated 0.852
%   a = 0.98: formula 2.104, simulated 2.180
fig = figure('Color', 'w', 'Position', [100 100 600 360]);
h1 = semilogy(poles, predicted, 'o-'); hold on; box on; grid on;
h2 = semilogy(poles, simulated, 's');
xlabel('pole a'); ylabel('output noise variance (units of q^2)');
legend([h1 h2], {'(q^2/12) / (1 - a^2)', 'simulation with rounding'}, 'Location', 'northwest');
print(fig, fullfile(figdir, 'ch15_noise_gain.png'), '-dpng', '-r150');
fprintf('  saved figures/ch15_noise_gain.png\n');

%% 4. Coefficient rounding: 4th-order filter with poles e^{-kT}, k = 1..4
T_COEF = 0.05;                        % sampling period (s); try 0.1 and 0.02
DECIMALS = 3;                         % decimals kept in each stored coefficient
fprintf('\n4. Coefficient rounding, poles e^(-kT), T = %g s, %d decimals\n', T_COEF, DECIMALS);
p = exp(-T_COEF*(1:4));
aDirect = poly(p);                    % direct form: one 4th-order polynomial
aRounded = round(aDirect*10^DECIMALS)/10^DECIMALS;
direct = roots(aRounded);
cascade = round(p*10^DECIMALS)/10^DECIMALS;   % first-order sections: coefficient = pole
fprintf('  exact poles        '); fprintf(' %.4f', p); fprintf('\n');
% expected: 0.9512 0.9048 0.8607 0.8187
fprintf('  A(1) exact %.3e, after rounding %.3e\n', polyval(aDirect, 1), polyval(aRounded, 1));
% expected: 1.172e-04, after rounding 1.000e-03
fprintf('  direct form poles  ');
for i = 1:4, fprintf(' %.4f%+.4fj', real(direct(i)), imag(direct(i))); end
fprintf('   max |z| = %.4f\n', max(abs(direct)));
% expected: 1.0117 +- 0.1167j, 0.7563 +- 0.1152j   max |z| = 1.0184
fprintf('  cascade poles      '); fprintf(' %.3f', cascade);
fprintf('   largest move = %.4f\n', max(abs(cascade - p)));
% expected: 0.951 0.905 0.861 0.819   largest move = 0.0003
fig = figure('Color', 'w', 'Position', [100 100 540 420]);
hold on; box on; grid on;
th = linspace(-0.4, 0.4, 200);
h1 = plot(cos(th), sin(th), 'Color', [0.5 0.5 0.5], 'LineWidth', 1);
h2 = plot(p, 0*p, 'x', 'MarkerSize', 9, 'LineWidth', 2);
h3 = plot(real(direct), imag(direct), 'x', 'MarkerSize', 9, 'LineWidth', 2);
h4 = plot(cascade, 0*cascade, '+', 'MarkerSize', 11, 'LineWidth', 1.5);
axis equal; xlim([0.6 1.1]); ylim([-0.2 0.2]);
legend([h1 h2 h3 h4], {'|z| = 1', 'exact', 'direct form, rounded', 'cascade, rounded'}, ...
       'FontSize', 8, 'Location', 'west');
print(fig, fullfile(figdir, 'ch15_coef_rounding.png'), '-dpng', '-r150');
fprintf('  saved figures/ch15_coef_rounding.png\n');

%% 5. Deadband and limit cycles: x(k+1) = Q[a x(k)], x(0) = 10
fprintf('\n5. Rounded first-order recursion x(k+1) = Q[a x(k)], x(0) = 10\n');
n = 15;
runNames = {'a = 0.92, rounding', 'a = -0.92, rounding', 'a = 0.92, truncation toward 0'};
runA = [0.92 -0.92 0.92];
runQ = {@round, @round, @fix};        % fix() truncates toward zero
runs = zeros(3, n);
for r = 1:3
    x = zeros(1, n);
    x(1) = 10;
    for k = 1:n-1
        x(k+1) = runQ{r}(runA(r)*x(k));
    end
    runs(r, :) = x;
    fprintf('  %-30s:', runNames{r}); fprintf(' %d', x); fprintf('\n');
end
% expected:
%   a = 0.92, rounding            : 10 9 8 7 6 6 6 6 6 6 6 6 6 6 6
%   a = -0.92, rounding           : 10 -9 8 -7 6 -6 6 -6 6 -6 6 -6 6 -6 6
%   a = 0.92, truncation toward 0 : 10 9 8 7 6 5 4 3 2 1 0 0 0 0 0
fprintf('  deadband for |a| = 0.92: |x| <= 1/(2(1 - 0.92)) = %.2f\n', 1/(2*0.08));
% expected: 6.25
fig = figure('Color', 'w', 'Position', [100 100 640 360]);
hold on; box on; grid on;
k = 0:n-1;
h = plot(k, 10*0.92.^k, 'Color', [0.5 0.5 0.5]);
dx = [0 0.2 -0.2];
mk = {'o', 's', '^'};
for r = 1:3
    h(end+1) = plot(k + dx(r), runs(r, :), mk{r});
end
plot([0 n-1], [0 0], 'k-', 'LineWidth', 0.6);
xlabel('k'); ylabel('x(k) / q');
legend(h, [{'no rounding'}, runNames], 'FontSize', 8);
print(fig, fullfile(figdir, 'ch15_limit_cycles.png'), '-dpng', '-r150');
fprintf('  saved figures/ch15_limit_cycles.png\n');

%% 6. Saturation and windup: plant 1/(s+1), digital PI, |v| <= M_MAX
T_PI = 0.1;  KP = 4;  KI = 4;  M_MAX = 1.2;  T_TRACK = 1;
fprintf('\n6. PI with |v| <= %g: plant 1/(s+1), T = %g s, Kp = Ki = %g\n', M_MAX, T_PI, KP);
modes = {'none', 'windup', 'back', 'clamp'};
modeNames = {'no limit', 'limit, plain PI', 'limit, back-calculation', 'limit, conditional integration'};
colors = {[0.5 0.5 0.5], [0.84 0.15 0.16], [0.12 0.47 0.71], [0.17 0.63 0.17]};
fig = figure('Color', 'w', 'Position', [100 100 1000 380]);
ax1 = subplot(1, 2, 1); hold on; box on; grid on;
ax2 = subplot(1, 2, 2); hold on; box on; grid on;
h1 = gobjects(0);  h2 = gobjects(0);
for i = 1:4
    [Cy, Iy] = pi_loop(modes{i}, 12, T_PI, KP, KI, M_MAX, T_TRACK);
    t = T_PI*(0:numel(Cy)-1);
    settle = T_PI*find(abs(Cy - 1) > 0.02, 1, 'last');
    fprintf('  %-32s: peak c = %.3f, largest I = %.2f, 2%% settling = %.1f s\n', ...
            modeNames{i}, max(Cy), max(Iy), settle);
    keep = t <= 8;
    h1(end+1) = plot(ax1, t(keep), Cy(keep), 'Color', colors{i});
    if i > 1
        h2(end+1) = plot(ax2, t(keep), Iy(keep), 'Color', colors{i});
    end
end
% expected:
%   no limit                        : peak c = 1.004, largest I = 1.02, 2% settling = 0.8 s
%   limit, plain PI                 : peak c = 1.176, largest I = 2.77, 2% settling = 6.2 s
%   limit, back-calculation         : peak c = 1.002, largest I = 1.01, 2% settling = 1.8 s
%   limit, conditional integration  : peak c = 1.000, largest I = 1.00, 2% settling = 3.3 s
plot(ax1, [0 8], [1 1], ':', 'Color', [0.5 0.5 0.5]);
plot(ax2, [0 8], [M_MAX M_MAX], '--', 'Color', [0.5 0.5 0.5]);
title(ax1, 'output c(kT)'); title(ax2, 'integrator state I(k)');
xlabel(ax1, 't (s)'); xlabel(ax2, 't (s)');
legend(ax1, h1, modeNames, 'FontSize', 8, 'Location', 'southeast');
legend(ax2, h2, modeNames(2:4), 'FontSize', 8, 'Location', 'southeast');
print(fig, fullfile(figdir, 'ch15_windup.png'), '-dpng', '-r150');
fprintf('  saved figures/ch15_windup.png\n');

fprintf('\nFigures are in %s\n', figdir);

%% Local functions
function [Phi, Gam] = zoh(A, B, T)
% Exact ZOH discretization: Phi = e^{AT}, Gamma = int_0^T e^{As} ds B.
n = size(A, 1);
M = zeros(n + 1);
M(1:n, 1:n) = A;
M(1:n, n+1) = B;
E = expm(M*T);
Phi = E(1:n, 1:n);
Gam = E(1:n, n+1);
end

function [A, B, C] = plant_with_prefilter(wf)
% State space of G_p(s) = 1/(s(s+1)), optionally followed by the Butterworth
% prefilter H_f(s) = wf^2/(s^2 + sqrt(2) wf s + wf^2) on the measurement.
if isempty(wf)
    A = [0 1; 0 -1];
    B = [0; 1];
    C = [1 0];
else
    A = [0 1 0 0;
         0 -1 0 0;
         0 0 0 1;
         wf^2 0 -wf^2 -sqrt(2)*wf];
    B = [0; 1; 0; 0];
    C = [0 0 1 0];
end
end

function L = loop_gain(T, w, K, wf, delay)
% K z^-delay times the pulse transfer function of ZOH, plant (and prefilter), at z = e^{jwT}.
[A, B, C] = plant_with_prefilter(wf);
[Phi, Gam] = zoh(A, B, T);
z = exp(1j*w*T);
G = C*((z*eye(size(A, 1)) - Phi)\Gam);
L = K*G/z^delay;
end

function pm = phase_margin(T, K, wf, delay)
% Crossover |L| = 1 below the Nyquist frequency, and the phase margin in degrees.
wcr = fzero(@(w) abs(loop_gain(T, w, K, wf, delay)) - 1, [1e-3, 0.999*pi/T]);
pm = 180 + angle(loop_gain(T, wcr, K, wf, delay))*180/pi;
if pm > 180, pm = pm - 360; end
end

function c = step_response(K, delay, n)
% c(k) of the unity-feedback loop with D(z) = K z^-delay, unit step reference.
% G(z) = (b1 z + b0)/((z - 1)(z - a)), a = e^-1 (T = 1 s).
a = exp(-1);
b1 = a;  b0 = 1 - 2*a;
c = zeros(1, n);
m = zeros(1, n);                      % control applied to the plant
for i = 1:n                           % i = k + 1
    ci = 0;
    if i >= 2, ci = ci + (1 + a)*c(i-1) + b1*m(i-1); end
    if i >= 3, ci = ci - a*c(i-2) + b0*m(i-2); end
    c(i) = ci;
    if delay                          % m(k) computed at k is written at k + 1
        if i >= 2, m(i) = K*(1 - c(i-1)); else, m(i) = 0; end
    else
        m(i) = K*(1 - c(i));
    end
end
end

function Kmax = largest_stable_gain(delayFraction)
% Largest K for D(z) = K when the output leaves delayFraction*T after the
% sample (T = 1). State space with the delayed input:
%   x(k+1) = Phi x + G0 m(k) + G1 m(k-1).
[A, B, C] = plant_with_prefilter([]);
T = 1;  tau = delayFraction;
Phi = zoh(A, B, T);
[P1, G0] = zoh(A, B, T - tau);
[~, Gt] = zoh(A, B, tau);
G1 = P1*Gt;
lo = 0;  hi = 5;
for it = 1:60                         % bisection on the spectral radius
    K = (lo + hi)/2;
    Acl = zeros(3);
    Acl(1:2, 1:2) = Phi - K*G0*C;
    Acl(1:2, 3) = G1;
    Acl(3, 1:2) = -K*C;
    if max(abs(eig(Acl))) < 1
        lo = K;
    else
        hi = K;
    end
end
Kmax = lo;
end

function [Cy, Iy, V] = pi_loop(mode, tEnd, T, KP, KI, MMAX, TTRACK)
% mode: 'none' (no limit), 'windup' (plain PI), 'back' (back-calculation),
% 'clamp' (conditional integration).
p = exp(-T);                          % exact ZOH model of 1/(s+1)
c = 0;  I = 0;
n = round(tEnd/T) + 1;
Cy = zeros(1, n);  Iy = zeros(1, n);  V = zeros(1, n);
for k = 1:n
    e = 1 - c;                        % unit step reference
    m = KP*e + I;                     % PI output, before the limit
    if strcmp(mode, 'none')
        v = m;
    else
        v = max(-MMAX, min(MMAX, m));
    end
    Cy(k) = c;  Iy(k) = I;  V(k) = v;
    if strcmp(mode, 'back')           % back-calculation anti-windup
        I = I + KI*T*e + (T/TTRACK)*(v - m);
    elseif strcmp(mode, 'clamp')      % conditional integration
        if abs(m) <= MMAX || m*e < 0
            I = I + KI*T*e;
        end
    else
        I = I + KI*T*e;
    end
    c = p*c + (1 - p)*v;              % plant update over one period
end
end
