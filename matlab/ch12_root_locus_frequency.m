%% Computers in the Loop, Chapter 12: Root Locus and Frequency Response of Sampled-Data Loops
% The running loop (ZOH + 1/(s(s+1)), T = 1 s, gain K, unity feedback): root
% locus, the gain for zeta = 0.5, the effect of T, the w-plane Bode plot and
% margins, the Nyquist plot, and the locus with one sample of delay.
% Milad Siami
%
% Sections:
%   1. G(z) by zero-order-hold discretization
%   2. root locus: the zero, the circle, breakaway, critical gain
%   3. the gain that gives zeta = 0.5
%   4. the critical gain for T = 0.1, 0.5, 1 s
%   5. the w-plane transfer function, its Bode plot, gain and phase margins
%   6. the Nyquist plot and the encirclement count Z = N + P
%   7. the root locus with one sample of computation delay, D(z) = K/z
%
% Run this file from the matlab/ folder. Figures go to matlab/figures/.

clear; close all;
here = fileparts(mfilename('fullpath'));
figdir = fullfile(here, 'figures');
if ~exist(figdir, 'dir'), mkdir(figdir); end

%% 1. The pulse transfer function G(z) = (1 - z^-1) Z[Gp(s)/s]
% With the Control System Toolbox: G = c2d(tf(1, [1 1 0]), 1, 'zoh')
T = 1;
[num, den] = zoh_model(T);
a = num(1);  b = num(2);              % G(z) = (a z + b)/(z^2 - (1+a) z + a)
fprintf('1. G(z) for T = 1 s\n');
fprintf('   numerator   %.4f z + %.4f      (e^-1 = %.4f, 1 - 2e^-1 = %.4f)\n', a, b, exp(-1), 1 - 2*exp(-1));
% expected: 0.3679 z + 0.2642
fprintf('   denominator z^2 %+.4f z %+.4f  = (z - 1)(z - %.4f)\n', den(2), den(3), exp(-1));
% expected: z^2 -1.3679 z +0.3679 = (z - 1)(z - 0.3679)

G = @(z) polyval(num, z)./polyval(den, z);
clpoles = @(K, nm, dn) roots(polyadd(dn, K*nm));   % roots of den(z) + K num(z)

%% 2. Root locus: zero, circle, breakaway, critical gain
% With the Control System Toolbox: rlocus(G)
z0 = -b/a;                            % the zero of G(z), equal to 2 - e
R = sqrt((1 - z0)*(a - z0));          % radius of the circle centered at the zero
% breakaway and break-in points: A'B - AB' = 0 with A = den, B = num
bpoly = polyadd(conv(polyder(den), num), -conv(den, polyder(num)));
zbz = sort(real(roots(bpoly)), 'descend');
zb = zbz(1);  zi = zbz(2);
Kof = @(s) -polyval(den, s)/polyval(num, s);
Kcr = fzero(@(K) max(abs(clpoles(K, num, den))) - 1, [1 5]);
pc = clpoles(Kcr, num, den);
theta = abs(angle(pc(1)));
fprintf('\n2. Root locus\n');
fprintf('   zero z0 = %.4f (2 - e = %.4f); circle radius R = %.4f\n', z0, 2 - exp(1), R);
% expected: z0 = -0.7183; R = 1.3661
fprintf('   breakaway %.4f at K = %.3f;  break-in %.4f at K = %.2f\n', zb, Kof(zb), zi, Kof(zi));
% expected: breakaway 0.6479 at K = 0.196;  break-in -2.0844 at K = 15.05
fprintf('   critical gain %.3f  ((e-1)/(e-2) = %.3f), crossing angle %.3f rad, %.2f samples per cycle\n', ...
        Kcr, (exp(1) - 1)/(exp(1) - 2), theta, 2*pi/theta);
% expected: critical gain 2.392, crossing angle 1.324 rad, 4.74 samples per cycle

%% 3. The gain for zeta = 0.5
K05 = fzero(@(K) zeta_of(first_root(clpoles(K, num, den))) - 0.5, [0.2 1]);
p05 = first_root(clpoles(K05, num, den));
[~, wn05] = zeta_of(p05);
fprintf('\n3. Target damping\n');
fprintf('   zeta = 0.5 at K = %.3f: poles %.3f +- j%.3f, wn = %.3f rad/s, ramp error 1/K = %.2f\n', ...
        K05, real(p05), abs(imag(p05)), wn05, 1/K05);
% expected: K = 0.516: poles 0.589 +- j0.397, wn = 0.685 rad/s, ramp error 1.94

% Figure: the root locus with the unit circle and the zeta = 0.5 spiral
fig = figure('Color', 'w', 'Position', [100 100 700 550]);
hold on; box on;
Ks = [linspace(0, 0.2, 400), linspace(0.2, 40, 4000)];
P = zeros(2, numel(Ks));
for i = 1:numel(Ks)
    P(:, i) = clpoles(Ks(i), num, den);
end
plot(real(P(:)), imag(P(:)), '.', 'MarkerSize', 3, 'Color', [0.86 0.08 0.24]);
tc = linspace(0, 2*pi, 400);
plot(cos(tc), sin(tc), 'Color', [0.5 0.5 0.5], 'LineWidth', 0.8);
ph = linspace(0, pi, 300);
cz = 0.5/sqrt(1 - 0.25);
plot(exp(-cz*ph).*cos(ph),  exp(-cz*ph).*sin(ph), '--', 'Color', [0.12 0.47 0.71], 'LineWidth', 0.8);
plot(exp(-cz*ph).*cos(ph), -exp(-cz*ph).*sin(ph), '--', 'Color', [0.12 0.47 0.71], 'LineWidth', 0.8);
plot([1 a], [0 0], 'kx', 'MarkerSize', 8, 'LineWidth', 2);
plot(z0, 0, 'ko', 'MarkerFaceColor', 'w', 'MarkerSize', 7);
plot(real(pc), imag(pc), 'o', 'Color', [0.86 0.08 0.24], 'MarkerFaceColor', [0.86 0.08 0.24]);
plot([real(p05) real(p05)], [imag(p05) -imag(p05)], 'o', 'Color', [0 0.5 0], 'MarkerFaceColor', [0 0.5 0]);
plot([-2.6 1.4], [0 0], 'k-', 'LineWidth', 0.5);
plot([0 0], [-1.6 1.6], 'k-', 'LineWidth', 0.5);
axis equal; xlim([-2.6 1.4]); ylim([-1.6 1.6]);
title('Root locus, ZOH + 1/(s(s+1)), T = 1 s'); xlabel('Re z'); ylabel('Im z');
print(fig, fullfile(figdir, 'ch12_root_locus.png'), '-dpng', '-r150');

%% 4. Effect of the sampling period on the critical gain
fprintf('\n4. Critical gain against T\n');
fprintf('   T      zero     Kcr     2/T+1/3   theta    theta/T\n');
for Ti = [0.1 0.5 1.0]
    [ni, di] = zoh_model(Ti);
    Kc = fzero(@(K) max(abs(clpoles(K, ni, di))) - 1, [0.5 100]);
    pci = clpoles(Kc, ni, di);
    thi = abs(angle(pci(1)));
    fprintf('   %-5g  %.3f  %7.2f  %7.2f   %.3f    %.2f\n', Ti, -ni(2)/ni(1), Kc, 2/Ti + 1/3, thi, thi/Ti);
end
% expected:
%   0.1    -0.967    20.34    20.33   0.444    4.44
%   0.5    -0.847     4.36     4.33   0.963    1.93
%   1      -0.718     2.39     2.33   1.324    1.32

%% 5. The w-plane: Gw(w) = G(z) with z = (1 + wT/2)/(1 - wT/2); Bode plot and margins
% With the Control System Toolbox: margin(G) gives the same margins (in omega).
Gw = @(w) G((1 + w*T/2)./(1 - w*T/2));
ww = logspace(-1.3, 2, 600);
resp = Gw(1j*ww);
magdB = 20*log10(abs(resp));
phase = unwrap(angle(resp))*180/pi;
wwc = fzero(@(x) abs(Gw(1j*x)) - 1, [0.1 1.5]);      % gain crossover
wwp = fzero(@(x) imag(Gw(1j*x)), [1 3]);             % phase crossover
PM = 180 + angle(Gw(1j*wwc))*180/pi;
GM = 1/abs(Gw(1j*wwp));
toOmega = @(x) (2/T)*atan(x*T/2);                     % undo the warping
fprintf('\n5. Margins at K = 1 (w-plane)\n');
fprintf('   pole at w = -%.4f (2 tanh(1/2) = %.4f), zeros at w = %.0f and w = -%.2f\n', ...
        2*(1 - a)/(1 + a), 2*tanh(0.5), 2/T, 2*(a + b)/(a - b));
% expected: pole at w = -0.9242, zeros at w = 2 and w = -12.20
fprintf('   PM = %.1f deg at ww = %.3f (omega = %.3f rad/s)\n', PM, wwc, toOmega(wwc));
% expected: PM = 30.4 deg at ww = 0.812 (omega = 0.772 rad/s)
fprintf('   GM = %.3f = %.2f dB at ww = %.3f (omega = %.3f rad/s)\n', GM, 20*log10(GM), wwp, toOmega(wwp));
% expected: GM = 2.392 = 7.58 dB at ww = 1.559 (omega = 1.324 rad/s)
fprintf('   GM equals the critical gain of the root locus: %d\n', abs(GM - Kcr) <= 1e-8 + 1e-5*abs(Kcr));
% expected: 1 (true)

fig = figure('Color', 'w', 'Position', [100 100 700 600]);
subplot(2, 1, 1);
h1 = semilogx(ww, magdB, 'Color', [0.86 0.08 0.24]); hold on; box on;
h2 = semilogx(ww, -20*log10(ww.*sqrt(1 + ww.^2)), '--', 'Color', [0.5 0.5 0.5]);
semilogx([ww(1) ww(end)], [0 0], 'k-', 'LineWidth', 0.6);
semilogx([wwc wwc], [-45 30], 'k:', 'LineWidth', 0.8);
semilogx([wwp wwp], [-45 30], 'k:', 'LineWidth', 0.8);
ylabel('magnitude (dB)'); ylim([-45 30]);
legend([h1 h2], {'sampled, G_w(j\omega_w)', 'analog G_p(j\omega)'});
title(sprintf('w-plane Bode plot, K = 1: PM = %.1f deg, GM = %.2f dB', PM, 20*log10(GM)));
subplot(2, 1, 2);
semilogx(ww, phase, 'Color', [0.86 0.08 0.24]); hold on; box on;
semilogx(ww, -90 - atand(ww), '--', 'Color', [0.5 0.5 0.5]);
semilogx([ww(1) ww(end)], [-180 -180], 'k-', 'LineWidth', 0.6);
yl = ylim;
semilogx([wwc wwc], yl, 'k:', 'LineWidth', 0.8);
semilogx([wwp wwp], yl, 'k:', 'LineWidth', 0.8);
ylabel('phase (deg)'); xlabel('\omega_w (rad/s)');
print(fig, fullfile(figdir, 'ch12_bode_w.png'), '-dpng', '-r150');

%% 6. Nyquist plot and Z = N + P
% The contour is the unit circle, counterclockwise, with a small outward
% half-circle around the pole at z = 1 (so that pole counts as inside: P = 0).
% With the Control System Toolbox: nyquist(G)
epsc = 1e-3;
t0 = 2*asin(epsc/2);
thc = linspace(t0, 2*pi - t0, 200000);
phc = linspace(angle(exp(-1j*t0) - 1), angle(exp(1j*t0) - 1), 20000);
gammaPath = [exp(1j*thc), 1 + epsc*exp(1j*phc)];
Lgam = G(gammaPath);
fprintf('\n6. Nyquist: P = 0 (the pole at z = 1 is indented and counts as inside)\n');
fprintf('   crossings of the negative real axis: %.3f and G(-1) = %.4f\n', -1/Kcr, G(-1));
% expected: -0.418 and G(-1) = -0.0379
for K = [1 10 30]
    N = clockwise_encirclements(Lgam, -1/K);
    Z = sum(abs(clpoles(K, num, den)) > 1);
    fprintf('   K = %5.1f: N = %d, Z = N + P = %d, poles outside the circle: %d\n', K, N, N, Z);
end
% expected: K = 1: N = 0, Z = 0;  K = 10: N = 2, Z = 2;  K = 30: N = 1, Z = 1

th = linspace(0.3, pi, 800);
Lh = G(exp(1j*th));
fig = figure('Color', 'w', 'Position', [100 100 550 650]);
hold on; box on;
h1 = plot(real(Lh), imag(Lh), 'Color', [0.86 0.08 0.24]);
h2 = plot(real(Lh), -imag(Lh), ':', 'Color', [0.86 0.08 0.24]);
plot(cos(tc), sin(tc), 'Color', [0.5 0.5 0.5], 'LineWidth', 0.8);
plot(-1, 0, 'k+', 'MarkerSize', 10, 'LineWidth', 2);
plot([-1.5 -1.5], [-2.2 2.2], '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 0.8);
plot([-2 1.1], [0 0], 'k-', 'LineWidth', 0.5);
plot([0 0], [-2.2 2.2], 'k-', 'LineWidth', 0.5);
axis equal; xlim([-2 1.1]); ylim([-2.2 2.2]);
legend([h1 h2], {'0 < \omega < \omega_s/2', 'mirror image'}, 'Location', 'northeast');
title('Nyquist plot of G(e^{j\omega T}), T = 1 s');
print(fig, fullfile(figdir, 'ch12_nyquist.png'), '-dpng', '-r150');

%% 7. One sample of computation delay: D(z) = K/z
denD = conv(den, [1 0]);              % z (z - 1)(z - a)
Kd = fzero(@(K) max(abs(clpoles(K, num, denD))) - 1, [0.3 2]);
pd = clpoles(Kd, num, denD);
fprintf('\n7. With one sample of delay\n');
fprintf('   asymptote center %.3f; critical gain %.3f; crossing angle %.3f rad; third pole %.3f\n', ...
        (1 + a - z0)/2, Kd, max(angle(pd(abs(imag(pd)) > 1e-9))), min(real(pd)));
% expected: asymptote center 1.043; critical gain 0.805; crossing angle 0.660 rad; third pole -0.213

fig = figure('Color', 'w', 'Position', [100 100 550 550]);
hold on; box on;
Ks = linspace(0, 8, 6000);
P = zeros(3, numel(Ks));
for i = 1:numel(Ks)
    P(:, i) = clpoles(Ks(i), num, denD);
end
plot(real(P(:)), imag(P(:)), '.', 'MarkerSize', 3, 'Color', [0.86 0.08 0.24]);
plot(cos(tc), sin(tc), 'Color', [0.5 0.5 0.5], 'LineWidth', 0.8);
plot([1 a 0], [0 0 0], 'kx', 'MarkerSize', 8, 'LineWidth', 2);
plot(z0, 0, 'ko', 'MarkerFaceColor', 'w', 'MarkerSize', 7);
xa = (1 + a - z0)/2;
plot([xa xa], [-1.4 1.4], '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 0.8);
plot([-1.2 1.5], [0 0], 'k-', 'LineWidth', 0.5);
plot([0 0], [-1.4 1.4], 'k-', 'LineWidth', 0.5);
axis equal; xlim([-1.2 1.5]); ylim([-1.4 1.4]);
title('Root locus with D(z) = K/z, T = 1 s');
print(fig, fullfile(figdir, 'ch12_root_locus_delay.png'), '-dpng', '-r150');

fprintf('\nFigures written to %s\n', figdir);

%% Local functions
function [num, den] = zoh_model(T)
% G(z) for a ZOH in front of 1/(s(s+1)): num = [a b], den = [1 -(1+a) a].
A = [0 1; 0 -1];  B = [0; 1];  C = [1 0];
M = zeros(3);
M(1:2, 1:2) = A;
M(1:2, 3) = B;
E = expm(M*T);
Phi = E(1:2, 1:2);
Gam = E(1:2, 3);
den = poly(Phi);
num = poly(Phi - Gam*C) - den;        % C(zI - Phi)^-1 Gam = num/den, num(1) = 0
num = num(2:end);
end

function p = polyadd(p1, p2)
% Sum of two polynomials of possibly different lengths.
n = max(numel(p1), numel(p2));
p = [zeros(1, n - numel(p1)), p1] + [zeros(1, n - numel(p2)), p2];
end

function r = first_root(rts)
% The root with the largest imaginary part (upper half-plane).
[~, i] = max(imag(rts));
r = rts(i);
end

function [zeta, wn] = zeta_of(z)
% Equivalent damping ratio and natural frequency (T = 1 s) of a z-plane pole.
r = abs(z);
th = abs(angle(z));
L = sqrt(log(r)^2 + th^2);
zeta = -log(r)/L;
wn = L;
end

function N = clockwise_encirclements(curve, q)
% Number of clockwise encirclements of the point q by the closed curve.
d = diff(angle([curve, curve(1)] - q));
d = mod(d + pi, 2*pi) - pi;
N = -round(sum(d)/(2*pi));
end
