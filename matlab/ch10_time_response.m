%% Computers in the Loop, Chapter 10: Closed-Loop Time Response and Steady-State Accuracy
% Examples 1, 2 and B, the effect of T, the gain limits of the laser loop,
% and the practice exercise; the output is also computed between the samples.
% Milad Siami
%
% Loop: R -> (+/-) -> sampler -> D(z) -> ZOH -> G_p(s) -> C, unity feedback.
% A z-plane pole r e^{j theta} has  zeta = -ln r / sqrt(ln^2 r + theta^2),
% wn = sqrt(ln^2 r + theta^2)/T  and  tau = -T/ln r.
%
% Run this file from the matlab/ folder. Figures go to matlab/figures/.

clear; close all;
here = fileparts(mfilename('fullpath'));
figdir = fullfile(here, 'figures');
if ~exist(figdir, 'dir'), mkdir(figdir); end

%% Example 1: G_p = 4/(s+2), T = 0.1 s
% With the Control System Toolbox: G = c2d(tf(4, [1 2]), 0.1); feedback(G, 1)
T = 0.1;
[nz, dz] = zoh_tf(4, [1 2], T);
[nc, dc] = closed_loop(nz, dz, 1);
pole = -dc(2);
[~, ~, tau] = specs(pole, T);
c = filter(nc, dc, ones(1, 6));
fprintf('Example 1: G_p = 4/(s+2), T = 0.1 s\n');
fprintf('  G(z) = %.4f/(z - %.4f);  C/R = %.4f/(z - %.4f)\n', nz(end), -dz(2), nc(end), pole);
% expected: G(z) = 0.3625/(z - 0.8187);  C/R = 0.3625/(z - 0.4562)
fprintf('  tau = -T/ln r = %.3f s (analog loop 4/(s+6): %.3f s);  equivalent s-pole %.3f\n', ...
        tau, 1/6, log(pole)/T);
% expected: tau = 0.127 s (analog 0.167 s);  equivalent s-pole -7.848
fprintf('  c(kT) ='); fprintf(' %.4f', c); fprintf('  -> 2/3\n');
% expected: 0 0.3625 0.5279 0.6034 0.6378 0.6535
fprintf('  closed-loop pole 3e^(-2T) - 2 < 0 for T > ln(1.5)/2 = %.3f s; < -1 for T > ln(3)/2 = %.3f s\n', ...
        log(1.5)/2, log(3)/2);
% expected: 0.203 s and 0.549 s

%% Example 2: G_p = 1/(s(s+1)), T = 1 s
T = 1;
[nz, dz] = zoh_tf(1, [1 1 0], T);
[nc, dc] = closed_loop(nz, dz, 1);
zp = roots(dc);
[~, iu] = max(imag(zp));
zp = zp(iu);                               % the pole in the upper half-plane
[zeta, wn, tau] = specs(zp, T);
c = filter(nc, dc, ones(1, 21));
[t, y] = hybrid_step(1, [1 1 0], T, 1, 21, 50);
fprintf('\nExample 2: G_p = 1/(s(s+1)), T = 1 s\n');
fprintf('  G(z) = (%.4f z + %.4f)/(z^2 - %.4f z + %.4f)\n', nz(2), nz(3), -dz(2), dz(3));
% expected: G(z) = (0.3679 z + 0.2642)/(z^2 - 1.3679 z + 0.3679)
fprintf('  characteristic polynomial z^2 - %.4f z + %.4f: poles %.3f +- j%.3f = %.3f at +-%.3f rad\n', ...
        -dc(2), dc(3), real(zp), imag(zp), abs(zp), angle(zp));
% expected: z^2 - 1.0000 z + 0.6321: poles 0.500 +- j0.618 = 0.795 at +-0.891 rad
fprintf('  zeta = %.3f, wn = %.3f rad/s, tau = %.2f s, Mp = %.1f%%, %.1f samples per cycle\n', ...
        zeta, wn, tau, 100*exp(-pi*zeta/sqrt(1 - zeta^2)), 2*pi/angle(zp));
% expected: zeta = 0.249, wn = 0.920 rad/s, tau = 4.36 s, Mp = 44.5%, 7.1 samples per cycle
fprintf('  c(k), k = 0..10:'); fprintf(' %.3f', c(1:11)); fprintf('\n');
% expected: 0 0.368 1 1.4 1.4 1.147 0.894 0.801 0.868 0.994 1.077
[ymax, iy] = max(y);
fprintf('  peak at the samples %.3f; true peak between samples %.3f at t = %.2f s\n', max(c), ymax, t(iy));
% expected: 1.400 at the samples; 1.449 at t = 3.46 s between them
kIn = find(abs(c - 1) > 0.02, 1, 'last');  % last sample outside 2% is k = kIn - 1
fprintf('  samples within 2%% from k = %d on\n', kIn);
% expected: k = 16
ya = step_tf(1, [1 1 1], linspace(0, 20, 20001));
fprintf('  analog loop 1/(s^2+s+1): overshoot %.0f%%\n', 100*(max(ya) - 1));
% expected: 16%

%% The effect of T on Example 2: z^2 + (T-2) z + (1 - T e^-T) = 0
fprintf('\nEffect of T on Example 2 (z^2 + (T-2) z + (1 - T e^-T) = 0)\n');
fprintf('   T     poles                    |z|     zeta   peak of samples\n');
for T = [0.1 0.5 1.0 2.0 4.0]
    [nz, dz] = zoh_tf(1, [1 1 0], T);
    [nc, dc] = closed_loop(nz, dz, 1);
    rts = roots(dc);
    rmax = max(abs(rts));
    cs = filter(nc, dc, ones(1, floor(60/T) + 1));
    if rmax < 1 && all(abs(imag(rts)) > 1e-9)
        [~, iu] = max(imag(rts));
        zp = rts(iu);
        re = real(zp);
        if abs(re) < 5e-4, re = 0; end             % print 0.000, not -0.000
        zstr = sprintf('%.2f', specs(zp, T));
        pstr = sprintf('%.3f +- j%.3f', re, imag(zp));
        pk = max(cs);
    else
        rs = sort(real(rts));
        zstr = ' --';
        pstr = sprintf('%.3f, %.3f', rs(1), rs(2));
        pk = Inf;
    end
    fprintf('  %3.1f   %-22s  %.3f   %-5s  %.2f\n', T, pstr, rmax, zstr, pk);
end
% expected:
%   0.1   0.950 +- j0.084         0.954   0.47   1.18
%   0.5   0.750 +- j0.366         0.835   0.37   1.29
%   1.0   0.500 +- j0.618         0.795   0.25   1.40
%   2.0   0.000 +- j0.854         0.854   0.10   1.73
%   4.0   -1.271, -0.729          1.271    --    Inf
Tlim = fzero(@(T) T*(1 + exp(-T)) - 4, [1 6]);
fprintf('  a real pole reaches z = -1 when T (1 + e^-T) = 4: T = %.2f s\n', Tlim);
% expected: 3.92 s

%% Example B: a digital integrator D = z/(z-1) on ZOH + 1/(s+1), T = ln 2
T = log(2);
fprintf('\nExample B: ZOH + 1/(s+1), T = ln 2, G(z) = 0.5/(z - 0.5)\n');
fprintf('  without D: c(k) ='); fprintf(' %.4f', filter([0 0.5], [1 0], ones(1, 6)));
fprintf('  (C/R = 0.5/z)\n');
% expected: 0 0.5 0.5 0.5 0.5 0.5
nc = [0 0.5 0];  dc = [1 -1 0.5];          % D G/(1 + D G) with D = z/(z-1)
c = filter(nc, dc, ones(1, 12));
zp = roots(dc);
zp = zp(1);
fprintf('  with D = z/(z-1): c(k) ='); fprintf(' %.4f', c(1:9)); fprintf('\n');
% expected: 0 0.5 1 1.25 1.25 1.125 1 0.9375 0.9375
fprintf('  poles 0.5 +- j0.5: r = %.3f, theta = pi/4, zeta = %.2f\n', abs(zp), specs(zp, T));
% expected: r = 0.707, zeta = 0.40
ramp = (0:399)*T;
e = ramp - filter(nc, dc, ramp);
fprintf('  ramp error with D: %.4f = T = ln 2\n', e(end));
% expected: 0.6931

%% The laser loop G_p = 2/((s+1)(s+4)) with D = k
fprintf('\nLaser loop, G_p = 2/((s+1)(s+4)), D = k\n');
for T = [0.1 1.0]
    [nz, dz] = zoh_tf(2, [1 5 4], T);
    f = @(kk) max(abs(roots(dz + kk*nz))) - 1;   % > 0 once a pole leaves the circle
    if T < 0.5, hi = 5000; else, hi = 50; end
    kmax = fzero(f, [0.5 hi]);
    fprintf('  T = %g: G(z) = (%.4f z + %.4f)/(z^2 - %.4f z + %.4f), stable for -2 < k < %.2f;  K_p = k G(1) = k x %.3f\n', ...
            T, nz(2), nz(3), -dz(2), dz(3), kmax, sum(nz)/sum(dz));
end
% expected:
%   T = 0.1: G(z) = (0.0085 z + 0.0072)/(z^2 - 1.5752 z + 0.6065), stable for -2 < k < 54.71; K_p = k x 0.500
%   T = 1:   G(z) = (0.2578 z + 0.0525)/(z^2 - 0.3862 z + 0.0067), stable for -2 < k < 6.78;  K_p = k x 0.500
fprintf('  5%% step error needs k >= %.0f; with T = 1 s the best is 2/(2 + 6.78) = %.2f\n', 2/0.05 - 2, 2/8.78);
% expected: k >= 38; best 0.23

%% Practice exercise: Example 2 with K = 2
T = 1;
[nz, dz] = zoh_tf(1, [1 1 0], T);
[nc, dc] = closed_loop(nz, dz, 2);
zp = roots(dc);
[~, iu] = max(imag(zp));
zp = zp(iu);
c = filter(nc, dc, ones(1, 60));
fprintf('\nPractice exercise, K = 2: z^2 - %.3f z + %.3f, poles %.3f +- j%.3f, |z| = %.3f, zeta = %.2f, peak of the samples %.2f, ramp error 1/K_v = 0.5\n', ...
        -dc(2), dc(3), real(zp), imag(zp), abs(zp), specs(zp, T), max(c));
% expected: z^2 - 0.632 z + 0.896, poles 0.316 +- j0.892, |z| = 0.947, zeta = 0.04, peak 1.73

%% Figure: Example 2 between the samples, and the samples for several T
fig = figure('Color', 'w', 'Position', [100 100 1000 340]);
subplot(1, 2, 1); hold on; box on;
T = 1;
[t, y] = hybrid_step(1, [1 1 0], T, 1, 21, 50);
[nz, dz] = zoh_tf(1, [1 1 0], T);
[nc, dc] = closed_loop(nz, dz, 1);
c = filter(nc, dc, ones(1, 21));
ta = linspace(0, 20, 400);
ya = step_tf(1, [1 1 1], ta);
h1 = plot(ta, ya, 'k--', 'LineWidth', 1);
h2 = plot(t, y, 'Color', [0.5 0.5 0.5]);
h3 = plot(0:20, c, 'o', 'MarkerSize', 4, 'Color', [0.84 0.15 0.16]);
xlabel('t (s)'); title('Example 2, T = 1 s');
legend([h1 h2 h3], {'analog loop', 'sampled loop c(t)', 'samples c(kT)'}, 'FontSize', 8, 'Location', 'southeast');
subplot(1, 2, 2); hold on; box on;
h = gobjects(0); labels = {};
for T = [0.1 0.5 1.0 2.0]
    [nz, dz] = zoh_tf(1, [1 1 0], T);
    [nc, dc] = closed_loop(nz, dz, 1);
    n = floor(25/T) + 1;
    h(end+1) = plot((0:n-1)*T, filter(nc, dc, ones(1, n)), '.-', 'MarkerSize', 6);
    labels{end+1} = sprintf('T = %g s', T);
end
xlabel('t (s)'); title('Example 2: the samples for several T');
legend(h, labels, 'FontSize', 8, 'Location', 'southeast');
print(fig, fullfile(figdir, 'ch10_time_response.png'), '-dpng', '-r150');
fprintf('\nSaved figures/ch10_time_response.png\n');

%% Local functions
function [nc, dc] = closed_loop(nz, dz, K)
% C/R = K G/(1 + K G) for G = nz/dz (same lengths).
nc = K*nz;
dc = dz + K*nz;
end

function [zeta, wn, tau] = specs(zp, T)
% zeta, wn, tau of a z-plane pole r e^{j theta}.
r = abs(zp);
th = abs(angle(zp));
L = sqrt(log(r)^2 + th^2);
zeta = -log(r)/L;
wn = L/T;
tau = -T/log(r);
end

function [t, y] = hybrid_step(num, den, T, D, n, sub)
% Unit-step response of the sampled loop with D = constant gain, between
% the samples too: the plant is integrated exactly on a grid of T/sub.
[A, B, C, ~] = tf2ss_ccf(num, den);
nx = size(A, 1);
M = zeros(nx + 1);
M(1:nx, 1:nx) = A;
M(1:nx, nx+1) = B;
E = expm(M*T/sub);
Ph = E(1:nx, 1:nx);
Ga = E(1:nx, nx+1);
x = zeros(nx, 1);
t = zeros(1, n*sub);
y = zeros(1, n*sub);
i = 0;
for k = 0:n-1
    u = D*(1 - C*x);                  % sample the error, hold the control
    for j = 0:sub-1
        i = i + 1;
        t(i) = (k + j/sub)*T;
        y(i) = C*x;
        x = Ph*x + Ga*u;
    end
end
end

function [numz, denz] = zoh_tf(num, den, T)
% ZOH equivalent of num(s)/den(s) as polynomials in z; the numerator is
% padded to the length of the denominator (leading zero kept).
[A, B, C, D] = tf2ss_ccf(num, den);
nx = size(A, 1);
M = zeros(nx + 1);
M(1:nx, 1:nx) = A;
M(1:nx, nx+1) = B;
E = expm(M*T);
Phi = E(1:nx, 1:nx);
Gam = E(1:nx, nx+1);
denz = poly(Phi);
numz = poly(Phi - Gam*C) - denz + D*denz;   % det(zI - Phi + Gam C) - det(zI - Phi)
end

function [A, B, C, D] = tf2ss_ccf(num, den)
% Controllable canonical form of num(s)/den(s) (deg num <= deg den).
num = num/den(1);
den = den/den(1);
nx = numel(den) - 1;
num = [zeros(1, nx + 1 - numel(num)), num];
D = num(1);
C = num(2:end) - D*den(2:end);
A = [-den(2:end); eye(nx - 1, nx)];
B = [1; zeros(nx - 1, 1)];
end

function y = step_tf(num, den, t)
% Unit-step response of num(s)/den(s) on the uniform grid t (t(1) = 0).
[A, B, C, D] = tf2ss_ccf(num, den);
nx = size(A, 1);
M = zeros(nx + 1);
M(1:nx, 1:nx) = A;
M(1:nx, nx+1) = B;
E = expm(M*(t(2) - t(1)));
Phi = E(1:nx, 1:nx);
Gam = E(1:nx, nx+1);
x = zeros(nx, 1);
y = zeros(size(t));
for k = 1:numel(t)
    y(k) = C*x + D;
    x = Phi*x + Gam;
end
end
