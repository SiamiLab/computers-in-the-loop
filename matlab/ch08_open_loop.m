%% Computers in the Loop, Chapter 8: Open-Loop Sampled-Data Systems
% E(z) of a sampled signal, the pulse transfer function of ZOH + 1/(s+1),
% where the samplers sit, a digital filter in front of the plant, and how
% large T may be for 25/(s^2 + 2s + 25).
% Milad Siami
%
% The ZOH equivalent is computed from one matrix exponential:
%   expm([A B; 0 0] T) = [Phi Gamma; 0 I],  G(z) = C (zI - Phi)^-1 Gamma + D.
% With the Control System Toolbox the same G(z) is  c2d(tf(num, den), T, 'zoh').
%
% Run this file from the matlab/ folder. Figures go to matlab/figures/.

clear; close all;
here = fileparts(mfilename('fullpath'));
figdir = fullfile(here, 'figures');
if ~exist(figdir, 'dir'), mkdir(figdir); end

%% E(z) of a sampled signal: E(s) = 1/((s+1)(s+2)), T = 0.5 s
T = 0.5;
a = exp(-T);  b = exp(-2*T);
k = 0:7;
fprintf('E(s) = 1/((s+1)(s+2)), T = 0.5 s\n');
fprintf('  E(z) = %.3f z/((z - %.3f)(z - %.3f))\n', a - b, a, b);
% expected: E(z) = 0.239 z/((z - 0.607)(z - 0.368))
seriesE = filter([0, a - b], conv([1 -a], [1 -b]), [1, zeros(1, 7)]);
fprintf('  samples e^-kT - e^-2kT :'); fprintf(' %.4f', exp(-k*T) - exp(-2*k*T)); fprintf('\n');
fprintf('  series of E(z)         :'); fprintf(' %.4f', seriesE); fprintf('\n');
% expected (both): 0 0.2387 0.2325 0.1733 0.1170 0.0753 0.0473 0.0293

%% ZOH + 1/(s+1)
[numz, denz] = zoh_tf(1, [1 1], T);
fprintf('\nZOH + 1/(s+1), T = 0.5 s\n');
fprintf('  exact discretization: G(z) = %.4f/(z%s);  (1 - e^-T)/(z - e^-T) = %.4f/(z - %.4f)\n', ...
        numz(end), sg(denz(2)), 1 - a, a);
% expected: G(z) = 0.3935/(z - 0.6065) both ways
c = zeros(1, 6);
for i = 1:5                          % brute force: integrate the plant over one period
    c(i+1) = a*c(i) + (1 - a)*1;
end
fprintf('  step samples (brute force):'); fprintf(' %.3f', c);
fprintf('   1 - e^-kT:'); fprintf(' %.3f', 1 - exp(-(0:5)*T)); fprintf('\n');
% expected (both): 0 0.393 0.632 0.777 0.865 0.918
fprintf('  DC gain G(1) = %.4f = G_p(0)\n', polyval(numz, 1)/polyval(denz, 1));
% expected: 1.0000

%% Where the samplers sit: G1 = 1/s, G2 = 1/(s+1), unit impulse sequence in
imp = [1, zeros(1, 5)];
ca = filter([1 0 0], conv([1 -1], [1 -a]), imp);         % (a) G1(z) G2(z)
cb = filter([0, 1 - a, 0], conv([1 -1], [1 -a]), imp);   % (b) G1G2(z)
fprintf('\nG1 = 1/s, G2 = 1/(s+1), T = 0.5 s, unit impulse sequence in\n');
fprintf('  (a) sampler between the blocks, G1(z)G2(z):'); fprintf(' %.3f', ca(1:4)); fprintf('\n');
% expected: 1 1.607 1.974 2.198
fprintf('  (b) no sampler between them,    G1G2(z)   :'); fprintf(' %.3f', cb(1:4)); fprintf('\n');
% expected: 0 0.393 0.632 0.777

%% A digital filter D(z) = (2z - 1)/z in front of ZOH + 1/(s+1)
fprintf('\nDigital filter D(z) = (2z - 1)/z before ZOH + 1/(s+1), unit step, T = 0.5 s\n');
m = filter([2 -1], 1, ones(1, 8));                       % m(k) = 2e(k) - e(k-1)
cf = filter([0, 1 - a], [1, -a], m);
fromFormula = [0, 1 - (2*a - 1)*a.^((1:4) - 1)];
fprintf('  with the filter   :'); fprintf(' %.3f', cf(1:5));
fprintf('   formula 1 - (2a-1) a^(k-1):'); fprintf(' %.3f', fromFormula); fprintf('\n');
% expected (both): 0 0.787 0.871 0.922 0.952
fprintf('  without it (D = 1):'); fprintf(' %.3f', filter([0, 1 - a], [1, -a], ones(1, 5))); fprintf('\n');
% expected: 0 0.393 0.632 0.777 0.865

%% How large can T be? G_p = 25/(s^2 + 2s + 25)
num = 25;  den = [1 2 25];
tt = linspace(0, 9, 90001);
yc = step_tf(num, den, tt);          % With the Control System Toolbox: step(tf(num, den), tt)
[ypk, ipk] = max(yc);
fprintf('\nG_p = 25/(s^2 + 2s + 25): zeta = 0.2, wn = 5 rad/s\n');
fprintf('  continuous step: peak %.3f at t = %.2f s, period 2 pi/wd = %.2f s\n', ...
        ypk, tt(ipk), 2*pi/(5*sqrt(1 - 0.04)));
% expected: peak 1.527 at t = 0.64 s, period 1.28 s

fig = figure('Color', 'w', 'Position', [100 100 1000 340]);
Tlist = [1.0 0.1];
for j = 1:2
    T = Tlist(j);
    [numz, denz] = zoh_tf(num, den, T);
    if T < 0.5, tEnd = 6; else, tEnd = 9; end
    n = round(tEnd/T) + 1;
    cs = filter([0, numz], denz, ones(1, n));   % step samples, one sample of delay from z^-1
    [cmax, imax] = max(cs);
    fprintf('  T = %g: G(z) = (%.4f z%s)/(z^2%s z%s), G(1) = %.4f\n', T, numz(1), sg(numz(2)), ...
            sg(denz(2)), sg(denz(3)), polyval(numz, 1)/polyval(denz, 1));
    fprintf('          samples'); fprintf(' %.3f', cs(1:4));
    fprintf(' ...; largest sample %.3f at k = %d (t = %.1f s)\n', cmax, imax - 1, (imax - 1)*T);
    subplot(1, 2, j); hold on; box on;
    keep = tt <= (n - 1)*T;
    h1 = plot(tt(keep), yc(keep), 'Color', [0.5 0.5 0.5]);
    h2 = plot((0:n-1)*T, cs, 'o', 'MarkerSize', 4, 'Color', [0.84 0.15 0.16]);
    xlabel('t (s)'); title(sprintf('T = %g s', T));
    legend([h1 h2], {'c(t)', 'c(kT)'}, 'FontSize', 8, 'Location', 'southeast');
end
% expected:
%   T = 1:   G(z) = (1.0055 z - 0.0067)/(z^2 - 0.1365 z + 0.1353), G(1) = 1.0000
%            samples 0 1.006 1.136 1.018 ...; largest sample 1.136 at k = 2 (t = 2.0 s)
%   T = 0.1: G(z) = (0.1147 z + 0.1072)/(z^2 - 1.5968 z + 0.8187), G(1) = 1.0000
%            samples 0 0.115 0.405 0.775 ...; largest sample 1.515 at k = 6 (t = 0.6 s)
print(fig, fullfile(figdir, 'ch08_how_large_T.png'), '-dpng', '-r150');
fprintf('\nSaved figures/ch08_how_large_T.png\n');

%% Local functions
function [numz, denz] = zoh_tf(num, den, T)
% ZOH equivalent of num(s)/den(s), as polynomials in z (leading zeros of
% the numerator removed). Uses det(zI - Phi + Gamma C) = det(zI - Phi)(1 + C(zI - Phi)^-1 Gamma).
[A, B, C, D] = tf2ss_ccf(num, den);
nx = size(A, 1);
M = zeros(nx + 1);
M(1:nx, 1:nx) = A;
M(1:nx, nx+1) = B;
E = expm(M*T);
Phi = E(1:nx, 1:nx);
Gam = E(1:nx, nx+1);
denz = poly(Phi);
numz = poly(Phi - Gam*C) - denz + D*denz;
numz = round(numz*1e14)/1e14;
numz = numz(find(abs(numz) > 0, 1):end);
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
% Unit-step response of num(s)/den(s) on the uniform grid t (t(1) = 0),
% exact at the grid points.
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

function s = sg(v)
% ' + 0.1234' or ' - 0.1234' for printing polynomials
if v < 0
    s = sprintf(' - %.4f', abs(v));
else
    s = sprintf(' + %.4f', abs(v));
end
end
