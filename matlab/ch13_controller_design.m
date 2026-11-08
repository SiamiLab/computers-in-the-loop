%% Computers in the Loop, Chapter 13: Designing Digital Controllers
% One lead compensator converted six ways, a lead designed directly in z,
% deadbeat designs with and without ripple, and a digital PID.
% Milad Siami
%
% The plant of Parts A and B is G_p(s) = 1/(s(s+1)) driven by a zero-order
% hold. Polynomials are coefficient vectors in descending powers of s or z.
% Figures (in matlab/figures/):
%   ch13_emulation.png     analog lead vs. its Tustin emulation at T = 0.2 s and T = 1 s
%   ch13_direct_lead.png   lead D(z) designed on the z-plane (T = 1 s), samples and c(t)
%   ch13_deadbeat.png      minimum-time deadbeat (ripple) and ripple-free deadbeat
%   ch13_pid.png           digital PID (positional form) on 4/((s+1)(s+4)), T = 0.05 s
%
% Run this file from the matlab/ folder.

clear; close all;
here = fileparts(mfilename('fullpath'));
figdir = fullfile(here, 'figures');
if ~exist(figdir, 'dir'), mkdir(figdir); end

% the running plant 1/(s(s+1)): states c and dc/dt
A_RUN = [0 1; 0 -1];
B_RUN = [0; 1];
C_RUN = [1 0];

%% Part A: emulation of Gc(s) = 4(s+1)/(s+2), T = 0.2 s, six conversions
% With the Control System Toolbox: c2d(tf([4 4], [1 2]), 0.2, 'tustin'),
% c2d(..., 'matched'), c2d(..., 'zoh'), and c2d(..., c2dOptions('Method',
% 'tustin', 'PrewarpFrequency', sqrt(2))) for the prewarped Tustin rule.
fprintf('=== Part A: emulation of Gc(s) = 4(s+1)/(s+2), T = 0.2 s ===\n');
numc = [4 4];  denc = [1 2];
T = 0.2;
names = {'forward Euler', 'backward Euler', 'Tustin', 'Tustin, prewarped', ...
         'matched pole-zero', 'ZOH equivalent'};
Dn = cell(1, 6);  Dd = cell(1, 6);
[Dn{1}, Dd{1}] = substitute(numc, denc, [1 -1], T);           % s = (z-1)/T
[Dn{2}, Dd{2}] = substitute(numc, denc, [1 -1], [T 0]);       % s = (z-1)/(Tz)
[Dn{3}, Dd{3}] = tustin(numc, denc, T, []);                   % s = (2/T)(z-1)/(z+1)
[Dn{4}, Dd{4}] = tustin(numc, denc, T, sqrt(2));              % prewarped at w1 = sqrt(2)
[Dn{5}, Dd{5}] = matched_pole_zero(numc, denc, T);
[Dn{6}, Dd{6}] = zoh_first_order(4, 1, 2, T);
[gn, gd] = pulse_tf_running(T);
for i = 1:6
    p = roots(polyadd(conv(Dd{i}, gd), conv(Dn{i}, gn)));
    [~, iu] = max(imag(p));
    pair = p(iu);
    [~, ck] = simulate_loop(Dn{i}, Dd{i}, A_RUN, B_RUN, C_RUN, T, 200, 1);
    fprintf('%-18s D(z) = %.4f(z - %.4f)/(z - %.4f)   poles %.4f +/- j%.4f, zeta %.2f, overshoot %.1f%%\n', ...
            names{i}, Dn{i}(1), -Dn{i}(2)/Dn{i}(1), -Dd{i}(2), real(pair), imag(pair), ...
            zeta_of(pair), 100*(max(ck) - 1));
end
% expected:
%   forward Euler      D(z) = 4.0000(z - 0.8000)/(z - 0.6000)   poles 0.7761 +/- j0.2984, zeta 0.45, overshoot 22.2%
%   backward Euler     D(z) = 3.4286(z - 0.8333)/(z - 0.7143)   poles 0.8162 +/- j0.3047, zeta 0.36, overshoot 28.5%
%   Tustin             D(z) = 3.6667(z - 0.8182)/(z - 0.6667)   poles 0.7993 +/- j0.3042, zeta 0.39, overshoot 25.7%
%   Tustin, prewarped  D(z) = 3.6648(z - 0.8171)/(z - 0.6648)   poles 0.7992 +/- j0.3036, zeta 0.40, overshoot 25.7%
%   matched pole-zero  D(z) = 3.6375(z - 0.8187)/(z - 0.6703)   poles 0.8011 +/- j0.3038, zeta 0.39, overshoot 26.0%
%   ZOH equivalent     D(z) = 4.0000(z - 0.8352)/(z - 0.6703)   poles 0.7874 +/- j0.3219, zeta 0.38, overshoot 25.3%

% The same Tustin recipe at T = 0.2 s and at T = 1 s
fig = figure('Color', 'w', 'Position', [100 100 700 360]);
hold on; box on;
t = linspace(0, 14, 1000);
h = plot(t, 1 - exp(-t).*(cos(sqrt(3)*t) + sin(sqrt(3)*t)/sqrt(3)), 'k--');
labels = {'analog loop'};
marks = {'o', 's'};
Tlist = [0.2 1.0];
for j = 1:2
    Ts = Tlist(j);
    [n1, d1] = tustin(numc, denc, Ts, []);
    [tk, ck, ~, tFine, cFine] = simulate_loop(n1, d1, A_RUN, B_RUN, C_RUN, Ts, floor(14/Ts), 10);
    hl = plot(tFine, cFine, 'LineWidth', 0.8);
    h(end+1) = plot(tk, ck, marks{j}, 'MarkerSize', 3, 'Color', get(hl, 'Color'));
    labels{end+1} = sprintf('Tustin, T = %g s', Ts);
    fprintf('Tustin at T = %g: peak of the samples %.3f\n', Ts, max(ck));
end
% expected: T = 0.2: 1.257;  T = 1: 1.820
plot([0 14], [1 1], 'Color', [0.5 0.5 0.5], 'LineWidth', 0.5);
xlabel('t (s)'); ylabel('c');
legend(h, labels, 'Location', 'southeast');
title('Emulation: the same analog lead at two sampling periods');
print(fig, fullfile(figdir, 'ch13_emulation.png'), '-dpng', '-r150');

% The hold's phase lag at crossover: about wc T/2
wc = sqrt(2*sqrt(5) - 2);
pm = 90 - atand(wc/2);
fprintf('crossover %.3f rad/s, analog PM %.1f deg\n', wc, pm);
% expected: crossover 1.572 rad/s, analog PM 51.8 deg
for Ts = [0.05 0.1 0.2 0.5 1.0]
    lag = wc*Ts/2*180/pi;
    fprintf('  T = %4g: hold lag %4.1f deg, PM estimate %4.1f deg\n', Ts, lag, pm - lag);
end
% expected: lags 2.3, 4.5, 9.0, 22.5, 45.0 deg; PM estimates 49.6, 47.3, 42.8, 29.3, 6.8 deg

%% Part B: a lead controller designed on the z-plane (T = 1 s)
% D(z) = Kc (z - e^-1)/z cancels the plant pole at e^-1. Then 1 + DG = 0 is
% z^2 + (K'-1) z + K' c = 0 with K' = Kc e^-1 and c = e - 2. Its complex roots
% lie on the circle |z + c| = sqrt(c(1+c)); intersect it with the zeta = 0.5 curve.
fprintf('\n=== Part B: direct design, T = 1 s ===\n');
T = 1;
[gn, gd] = pulse_tf_running(T);
c = exp(1) - 2;                       % the plant zero is at -c
theta = fzero(@(th) exp(-th/sqrt(3)) - (-c*cos(th) + sqrt(c^2*cos(th)^2 + c)), [0.1 2.5]);
r = exp(-theta/sqrt(3));
Kprime = r^2/c;                       % the product of the roots is K'c
Kc = Kprime*exp(1);
fprintf('design pole %.4f +/- j%.4f, Kc = %.3f, Kv = %.3f\n', r*cos(theta), r*sin(theta), Kc, Kprime*(exp(1) - 1));
% expected: design pole 0.2870 +/- j0.4730, Kc = 1.158, Kv = 0.732
DnB = [Kc, -Kc*exp(-1)];  DdB = [1 0];
[tk, ck, uk, tFine, cFine] = simulate_loop(DnB, DdB, A_RUN, B_RUN, C_RUN, T, 15, 20);
fprintf('c(k):'); fprintf(' %.4f', ck(1:11)); fprintf('\n');
% expected: 0 0.4261 0.9766 1.1622 1.1003 1.0079 0.9738 0.9826 0.9980 1.0042 1.0030
fprintf('u(k):'); fprintf(' %.4f', uk(1:8)); fprintf('\n');
% expected: 1.1582 0.2386 -0.2175 -0.1978 -0.0470 0.0336 0.0337 0.0090
fig = figure('Color', 'w', 'Position', [100 100 700 340]);
hold on; box on;
h1 = plot(tFine, cFine, 'r-', 'LineWidth', 1);
h2 = plot(tk, ck, 'ro', 'MarkerSize', 4);
h3 = stairs(tk, uk, 'Color', [0.12 0.47 0.71], 'LineWidth', 1);
plot([0 tk(end)], [1 1], 'Color', [0.5 0.5 0.5], 'LineWidth', 0.5);
xlabel('t (s)');
legend([h1 h2 h3], {'c(t)', 'c(kT)', 'u (held)'});
title('Lead controller 1.158(z - 0.3679)/z, T = 1 s');
print(fig, fullfile(figdir, 'ch13_direct_lead.png'), '-dpng', '-r150');

%% Deadbeat: minimum time (ripple) and ripple-free
% Minimum time, F(z) = 1/z:                 D(z) = (e z - 1)/(z + c)
% Ripple-free, F(z) = (z + c)/((e-1) z^2):  D(z) = e/(e-1) (z - e^-1)/(z + (e-2)/(e-1))
dbNames = {'minimum time, F = 1/z', 'ripple-free, F = (z + c)/((e-1) z^2)'};
dbNum = {[exp(1), -1], exp(1)/(exp(1) - 1)*[1, -exp(-1)]};
dbDen = {[1, c], [1, (exp(1) - 2)/(exp(1) - 1)]};
fig = figure('Color', 'w', 'Position', [100 100 700 520]);
ax1 = subplot(2, 1, 1); hold on; box on;
ax2 = subplot(2, 1, 2); hold on; box on;
h = gobjects(0);
for i = 1:2
    [tk, ck, uk, tFine, cFine] = simulate_loop(dbNum{i}, dbDen{i}, A_RUN, B_RUN, C_RUN, T, 12, 40);
    fprintf('%s: c(k) =', dbNames{i}); fprintf(' %.4f', ck(1:4));
    fprintf(', u(k) ='); fprintf(' %.4f', uk(1:4));
    fprintf(', peak of c(t) = %.3f\n', max(cFine));
    h(end+1) = plot(ax1, tFine, cFine, 'LineWidth', 1);
    plot(ax1, tk, ck, 'o', 'MarkerSize', 3, 'Color', get(h(end), 'Color'));
    stairs(ax2, tk, uk, 'Color', get(h(end), 'Color'), 'LineWidth', 1);
end
% expected:
%   minimum time: c(k) = 0 1 1 1, u(k) = 2.7183 -2.9525 2.1207 -1.5233, peak of c(t) = 1.364
%   ripple-free:  c(k) = 0 0.5820 1 1, u(k) = 1.5820 -0.5820 0 0, peak of c(t) = 1.000

% The ripple-free controller from Ragazzini's formula D = F/(G(1 - F)),
% before cancelling common factors
[DnR, DdR] = ragazzini([1, c]/(exp(1) - 1), [1 0 0], gn, gd);
fprintf('ripple-free D(z), uncancelled form: zeros'); fprintf(' %.4f', sort(real(roots(DnR))));
fprintf('  poles'); fprintf(' %.4f', sort(real(roots(DdR)))); fprintf('\n');
% expected: zeros -0.7183 0.3679 1;  poles -0.7183 -0.4180 1
plot(ax1, [0 tk(end)], [1 1], 'Color', [0.5 0.5 0.5], 'LineWidth', 0.5);
ylabel(ax1, 'c'); title(ax1, 'Deadbeat for ZOH + 1/(s(s+1)), T = 1 s');
legend(ax1, h, dbNames, 'FontSize', 8, 'Location', 'southeast');
ylabel(ax2, 'u'); xlabel(ax2, 't (s)');
print(fig, fullfile(figdir, 'ch13_deadbeat.png'), '-dpng', '-r150');

%% Digital PID, positional form, on G_p = 4/((s+1)(s+4))
% integral by backward Euler, derivative filtered with time constant tau_f
fprintf('\n=== Digital PID ===\n');
Kp = 4;  Ki = 4;  Kd = 0.75;  T = 0.05;  tauf = 0.05;
alphaD = tauf/(tauf + T);
gD = Kd/(tauf + T);
Ap = [-1 1; 0 -4];                    % x1 = c, x2 = output of 4/(s+4)
Bp = [0; 4];
Cp = [1 0];
[Phi, Gam] = zoh_matrices(Ap, Bp, T);
x = zeros(2, 1);
uI = 0;  uD = 0;  eOld = 0;
nS = floor(4/T) + 1;
ck = zeros(1, nS);  uk = zeros(1, nS);
for i = 1:nS
    cNow = Cp*x;
    e = 1 - cNow;
    uI = uI + Ki*T*e;                 % integral, backward Euler
    uD = alphaD*uD + gD*(e - eOld);    % filtered derivative, backward Euler
    u = Kp*e + uI + uD;               % positional form
    eOld = e;
    ck(i) = cNow;
    uk(i) = u;
    x = Phi*x + Gam*u;
end
[cmax, imax] = max(ck);
fprintf('u(0) = %.2f; peak %.4f at t = %.2f s\n', uk(1), cmax, (imax - 1)*T);
% expected: u(0) = 11.70; peak 1.0156 at t = 1.60 s
t = linspace(0, 4, 400);
fig = figure('Color', 'w', 'Position', [100 100 700 340]);
hold on; box on;
h1 = plot(t, 1 - exp(-2*t) + t.*exp(-2*t), 'k--');
h2 = plot((0:nS-1)*T, ck, 'r.', 'MarkerSize', 8);
plot([0 4], [1 1], 'Color', [0.5 0.5 0.5], 'LineWidth', 0.5);
xlabel('t (s)'); ylabel('c');
legend([h1 h2], {'analog ideal PID', 'digital PID samples'}, 'Location', 'southeast');
title('PID: Kp = 4, Ki = 4, Kd = 0.75, filter 0.05 s, T = 0.05 s');
print(fig, fullfile(figdir, 'ch13_pid.png'), '-dpng', '-r150');

fprintf('\nplots saved in %s\n', figdir);

%% Local functions
function [numz, denz] = substitute(nums, dens, top, bot)
% Replace s by top(z)/bot(z) in nums/dens and clear the fractions.
% top and bot are polynomials in z of degree <= 1.
%   forward Euler:  s = (z-1)/T       top = [1 -1],        bot = T
%   backward Euler: s = (z-1)/(Tz)    top = [1 -1],        bot = [T 0]
%   Tustin:         s = c (z-1)/(z+1) top = [c -c],        bot = [1 1]
n = max(numel(nums), numel(dens)) - 1;
numz = convert(nums, n, top, bot);
denz = convert(dens, n, top, bot);
numz = numz/denz(1);
denz = denz/denz(1);
end

function out = convert(p, n, top, bot)
% sum_k p_k top^k bot^(n-k), the polynomial p(s) with s = top/bot, times bot^n.
p = [zeros(1, n + 1 - numel(p)), p];
out = 0;
for i = 1:n+1
    k = n - i + 1;                    % p(i) multiplies s^k
    term = p(i);
    for j = 1:k
        term = conv(term, top);
    end
    for j = 1:n-k
        term = conv(term, bot);
    end
    out = polyadd(out, term);
end
end

function [numz, denz] = tustin(nums, dens, T, w1)
% Tustin rule; with a prewarping frequency w1 the factor 2/T becomes w1/tan(w1 T/2).
if isempty(w1)
    c = 2/T;
else
    c = w1/tan(w1*T/2);
end
[numz, denz] = substitute(nums, dens, [c -c], [1 1]);
end

function [numz, denz] = matched_pole_zero(nums, dens, T)
% Map poles and zeros by z = e^{sT}, zeros at infinity to z = -1, match the DC gain.
zs = roots(nums);
ps = roots(dens);
zd = [exp(zs*T); -ones(numel(ps) - numel(zs), 1)];
numz = real(poly(zd));
denz = real(poly(exp(ps*T)));
gain = (polyval(nums, 0)/polyval(dens, 0))/(polyval(numz, 1)/polyval(denz, 1));
numz = gain*numz;
end

function [Phi, Gam] = zoh_matrices(A, B, T)
% Phi = e^{AT} and Gamma = int_0^T e^{A tau} d tau B, from one matrix exponential.
n = size(A, 1);
M = zeros(n + 1);
M(1:n, 1:n) = A;
M(1:n, n+1) = B;
E = expm(M*T);
Phi = E(1:n, 1:n);
Gam = E(1:n, n+1);
end

function [numz, denz] = zoh_first_order(K, a, b, T)
% ZOH (step-invariant) equivalent of K(s+a)/(s+b) = K + K(a-b)/(s+b).
% State space: x' = -b x + e, u = K(a-b) x + K e.
[phi, gam] = zoh_matrices(-b, 1, T);
numz = [K, -K*phi + K*(a - b)*gam];   % D(z) = K + K(a-b) gam/(z - phi)
denz = [1, -phi];
end

function [num, den] = pulse_tf_running(T)
% G(z) = (b1 z + b0)/((z-1)(z-e^{-T})) for ZOH + 1/(s(s+1)).
p = exp(-T);
num = [T - 1 + p, 1 - p - T*p];
den = conv([1 -1], [1 -p]);
end

function [tk, ck, uk, tFine, cFine] = simulate_loop(Dn, Dd, A, B, C, T, nSamples, fine)
% Unit-step response of the sampled loop r -> e -> D(z) -> ZOH -> plant -> c.
% The controller runs as a difference equation; the plant is propagated
% exactly between samples with matrix exponentials, so c(t) between the
% samples is exact too.
Dn = [zeros(1, numel(Dd) - numel(Dn)), Dn]/Dd(1);
Dd = Dd/Dd(1);
order = numel(Dd) - 1;
PhiJ = cell(1, fine);  GamJ = cell(1, fine);
for j = 1:fine
    [PhiJ{j}, GamJ{j}] = zoh_matrices(A, B, T*(j - 1)/fine);
end
[Phi, Gam] = zoh_matrices(A, B, T);
x = zeros(size(A, 1), 1);
eHist = zeros(1, order + 1);          % e(k), e(k-1), ...
uHist = zeros(1, order + 1);          % u(k), u(k-1), ...
ck = zeros(1, nSamples);  uk = zeros(1, nSamples);
tFine = zeros(1, nSamples*fine);  cFine = zeros(1, nSamples*fine);
i = 0;
for k = 1:nSamples
    eHist = [1 - C*x, eHist(1:end-1)];
    u = Dn*eHist.' - Dd(2:end)*uHist(1:end-1).';
    uHist = [u, uHist(1:end-1)];
    ck(k) = C*x;
    uk(k) = u;
    for j = 1:fine
        i = i + 1;
        tFine(i) = (k - 1 + (j - 1)/fine)*T;
        cFine(i) = C*(PhiJ{j}*x + GamJ{j}*u);
    end
    x = Phi*x + Gam*u;
end
tk = (0:nSamples-1)*T;
end

function z = zeta_of(p)
% Damping ratio of a z-plane pole (through s = ln(z)/T).
s = log(p);
z = -real(s)/abs(s);
end

function [Dn, Dd] = ragazzini(Fnum, Fden, gn, gd)
% D = F/(G(1 - F)) for F = Fnum/Fden, as polynomials (no cancellation is done).
oneMinusF = polyadd(Fden, -Fnum);
Dn = conv(Fnum, gd);
Dd = conv(gn, oneMinusF);
end

function p = polyadd(p1, p2)
% Sum of two polynomials of possibly different lengths.
n = max(numel(p1), numel(p2));
p = [zeros(1, n - numel(p1)), p1] + [zeros(1, n - numel(p2)), p2];
end
