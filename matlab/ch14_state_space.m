%% Computers in the Loop, Chapter 14: State-Space Design: Pole Placement, Observers, and Optimal Control
% Phi and Gamma of the running plant, controllability and observability,
% pole placement by Ackermann's formula, deadbeat feedback, reference gain
% and integral action, prediction and current observers, the
% controller-observer as D(z), and discrete LQR by Riccati iteration.
% Milad Siami
%
% The running example is a ZOH followed by G_p(s) = 1/(s(s+1)), sampled at
% T = 1 s, with states x1 = c (position) and x2 = dc/dt (velocity).
% Figures (in matlab/figures/): ch14_pole_placement.png, ch14_integral_action.png,
% ch14_observer.png, ch14_lqr.png.
%
% Run this file from the matlab/ folder.

clear; close all;
here = fileparts(mfilename('fullpath'));
figdir = fullfile(here, 'figures');
if ~exist(figdir, 'dir'), mkdir(figdir); end

%% 1. The plant and its exact sampled model
% With the Control System Toolbox: sysd = c2d(ss(A, B, C, 0), T); sysd.A, sysd.B
A = [0 1; 0 -1];
B = [0; 1];
C = [1 0];
T = 1;
[Phi, Gam] = sample_plant(A, B, T);
a = exp(-T);
fprintf('1. Sampled model\n');
fprintf('%28s = [%.4f %.4f; %.4f %.4f]\n', 'Phi', Phi(1,1), Phi(1,2), Phi(2,1), Phi(2,2));
% expected: [1 0.6321; 0 0.3679]
fprintf('%28s = [%.4f; %.4f]\n', 'Gamma', Gam(1), Gam(2));
% expected: [0.3679; 0.6321]
% compare with the closed forms found by hand
assert(max(max(abs(Phi - [1, 1 - a; 0, a]))) < 1e-12 && max(abs(Gam - [T - 1 + a; 1 - a])) < 1e-12);
% pulse transfer function G(z) = C (zI - Phi)^-1 Gamma at one test point
z0 = 0.3 + 0.7j;
Gstate = C*((z0*eye(2) - Phi)\Gam);
Gtable = (a*z0 + 1 - 2*a)/((z0 - 1)*(z0 - a));
fprintf('%28s = %s  %s\n', 'G(z0) two ways', cstr(Gstate, 6), cstr(Gtable, 6));
% expected: -0.627534+0.180331j (both)

%% 2. Controllability and observability
% With the Control System Toolbox: ctrb(Phi, Gam), obsv(Phi, C)
fprintf('\n2. Rank tests\n');
fprintf('%28s = %.4f\n', 'det Ctrb', det(ctrb_matrix(Phi, Gam)));
% expected: -0.3996
fprintf('%28s = %.4f\n', 'det Obsv', det(obsv_matrix(Phi, C)));
% expected: 0.6321
% the oscillator 1/(s^2 + 1) sampled at T = pi loses controllability
Ao = [0 1; -1 0];
for To = [pi/2, pi]
    [Po, Go] = sample_plant(Ao, B, To);
    fprintf('%28s = %d\n', sprintf('oscillator rank Ctrb, T=%.3f', To), rank(ctrb_matrix(Po, Go), 1e-9));
end
% expected: T = 1.571: 2;  T = 3.142: 1

%% 3. Pole placement and deadbeat
% With the Control System Toolbox: K = acker(Phi, Gam, [0.5+0.5j, 0.5-0.5j])
fprintf('\n3. State feedback\n');
alphaC = [1 -1 0.5];                  % (z - 0.5 - 0.5j)(z - 0.5 + 0.5j)
K = ackermann(Phi, Gam, alphaC);
fprintf('%28s = [%.4f %.4f]\n', 'K (poles 0.5 +- j0.5)', K(1), K(2));
% expected: [0.7910 0.1216]
ev = eig(Phi - Gam*K);
fprintf('%28s = %s, %s\n', 'eig(Phi - Gamma K)', cstr(ev(1), 4), cstr(ev(2), 4));
% expected: 0.5000+0.5000j, 0.5000-0.5000j (either order)
Kdb = ackermann(Phi, Gam, [1 0 0]);
fprintf('%28s = [%.4f %.4f]\n', 'deadbeat K', Kdb(1), Kdb(2));
% expected: [1.5820 1.2433]
Mdb = (Phi - Gam*Kdb)^2;
fprintf('%28s = %.1e\n', 'max |(Phi - Gamma Kdb)^2|', max(abs(Mdb(:))));
% expected: 0 up to round-off (about 1e-16)

fig = figure('Color', 'w', 'Position', [100 100 1000 340]);
ax1 = subplot(1, 2, 1); hold on; box on; grid on;
ax2 = subplot(1, 2, 2); hold on; box on; grid on;
gains = {K, Kdb};
labs = {'poles 0.5 +- j0.5', 'deadbeat'};
h1 = gobjects(0);  h2 = gobjects(0);
for i = 1:2
    [tc, cc, mseq] = regulate(Phi, Gam, gains{i}, [1; 0], 8, T);
    h1(end+1) = plot(ax1, tc, cc);
    plot(ax1, tc(1:20:end), cc(1:20:end), 'o', 'MarkerSize', 3, 'Color', get(h1(end), 'Color'));
    h2(end+1) = stairs(ax2, 0:8, [mseq, mseq(end)]);
end
xlabel(ax1, 't (s)'); ylabel(ax1, 'c(t)'); title(ax1, 'regulation from x(0) = [1, 0]');
xlabel(ax2, 't (s)'); ylabel(ax2, 'm(t)'); title(ax2, 'control (held by the ZOH)');
legend(ax1, h1, labs); legend(ax2, h2, labs);
print(fig, fullfile(figdir, 'ch14_pole_placement.png'), '-dpng', '-r150');

% deadbeat effort grows like 1/T^2
fprintf('   deadbeat effort for several T (x(0) = [1, 0]):\n');
for Ts = [2.0 1.0 0.5 0.2 0.1]
    [P_, G_] = sample_plant(A, B, Ts);
    Kt = ackermann(P_, G_, [1 0 0]);
    fprintf('      T = %4.1f   K = [%.3f %.3f]   m(0) = %9.3f\n', Ts, Kt(1), Kt(2), -Kt(1));
end
% expected:
%   T = 2.0   K = [0.578 0.554]     m(0) =    -0.578
%   T = 1.0   K = [1.582 1.243]     m(0) =    -1.582
%   T = 0.5   K = [5.083 2.707]     m(0) =    -5.083
%   T = 0.2   K = [27.583 7.183]    m(0) =   -27.583
%   T = 0.1   K = [105.083 14.675]  m(0) =  -105.083

%% 4. Reference gain and integral action
fprintf('\n4. Tracking\n');
Nbar = 1/(C*((eye(2) - Phi + Gam*K)\Gam));
fprintf('%28s = %.4f\n', 'Nbar', Nbar);
% expected: 0.7910
% augmented model: x_I(k+1) = x_I(k) + r(k) - c(k)
PhiA = [Phi, zeros(2, 1); -C, 1];
GamA = [Gam; 0];
Ka = ackermann(PhiA, GamA, conv(alphaC, [1 -0.5]));   % third pole at 0.5
k1 = Ka(1);  k2 = Ka(2);  kI = -Ka(3);
fprintf('%28s = [%.4f %.4f %.4f]\n', '[k1, k2, kI]', k1, k2, kI);
% expected: [1.3518 0.5862 0.3955]

N = 31;
w = zeros(1, N);
w(16:end) = 0.2;                      % constant input disturbance from k = 15
cNbar = zeros(1, N);  cInt = zeros(1, N);
x = zeros(2, 1);
for i = 1:N
    cNbar(i) = x(1);
    m = -K*x + Nbar*1;
    x = Phi*x + Gam*(m + w(i));
end
xa = zeros(3, 1);
for i = 1:N
    cInt(i) = xa(1);
    m = -Ka*xa;
    xa = PhiA*xa + GamA*(m + w(i)) + [0; 0; 1];   % r = 1 enters the integrator
end
fprintf('%28s = %.3f\n', 'final c, Nbar only', cNbar(end));
% expected: 1.252
fprintf('%28s = %.3f\n', 'final c, integral action', cInt(end));
% expected: 0.998

fig = figure('Color', 'w', 'Position', [100 100 800 320]);
hold on; box on; grid on;
h1 = plot(0:N-1, cNbar, 'o-', 'MarkerSize', 3);
h2 = plot(0:N-1, cInt, 's-', 'MarkerSize', 3);
plot([0 N-1], [1 1], ':', 'Color', [0.5 0.5 0.5]);
plot([15 15], ylim, '--', 'Color', [0.5 0.5 0.5]);
xlabel('k'); ylabel('c(k)'); title('unit step; +0.2 added at the plant input from k = 15');
legend([h1 h2], {'Nbar only', 'integral action'}, 'Location', 'southeast');
print(fig, fullfile(figdir, 'ch14_integral_action.png'), '-dpng', '-r150');

%% 5. Observers, poles at 0.2, 0.2
% Prediction observer: xhat(k+1) = Phi xhat + Gam m + L (c - C xhat).
% Current observer: the same formula with C Phi in place of C.
fprintf('\n5. Observers, poles at 0.2, 0.2\n');
alphaO = [1 -0.4 0.04];
L = observer_gain(Phi, C, alphaO);
Lc = observer_gain(Phi, C*Phi, alphaO);
fprintf('%28s = [%.4f; %.4f]\n', 'prediction L', L(1), L(2));
% expected: [0.9679; 0.0446]
fprintf('%28s = [%.4f; %.4f]\n', 'current Lc', Lc(1), Lc(2));
% expected: [0.8913; 0.1212]
fprintf('%28s = %.4f, %.4f\n', 'eig(Phi - L C)', real(eig(Phi - L*C)));
% expected: 0.2000, 0.2000
fprintf('%28s = %.4f, %.4f\n', 'eig(Phi - Lc C Phi)', real(eig(Phi - Lc*C*Phi)));
% expected: 0.2000, 0.2000

% separation: feed back the estimate, compare with true-state feedback
steps = 11;
x = [1; 0];  xh = [0; 0];  xs = [1; 0];
cObs = zeros(1, steps);  cSF = zeros(1, steps);  xh1 = zeros(1, steps);
for i = 1:steps
    cObs(i) = x(1);
    cSF(i) = xs(1);
    xh1(i) = xh(1);
    m = -K*xh;
    mSF = -K*xs;
    cMeas = C*x;
    xh = Phi*xh + Gam*m + L*(cMeas - C*xh);
    x = Phi*x + Gam*m;
    xs = Phi*xs + Gam*mSF;
end
big = [Phi, -Gam*K; L*C, Phi - Gam*K - L*C];
ev = eig(big);
fprintf('%28s =', 'closed-loop poles (x, xhat)');
for i = 1:4, fprintf(' %s', cstr(ev(i), 4)); end
fprintf('\n');
% expected: 0.2, 0.2, 0.5 +- 0.5j (any order)

fig = figure('Color', 'w', 'Position', [100 100 800 320]);
hold on; box on; grid on;
h1 = plot(0:steps-1, cSF, 'o-', 'MarkerSize', 3);
h2 = plot(0:steps-1, cObs, 's-', 'MarkerSize', 3);
h3 = plot(0:steps-1, xh1, '^--', 'MarkerSize', 3);
xlabel('k'); title('observer-based regulator, xhat(0) = 0');
legend([h1 h2 h3], {'c(k), m = -K x', 'c(k), m = -K xhat', 'estimate xhat1(k)'});
print(fig, fullfile(figdir, 'ch14_observer.png'), '-dpng', '-r150');

%% 6. The controller-observer as a transfer function D(z) = K (zI - Phi + Gam K + L C)^-1 L
fprintf('\n6. D(z) = K (zI - Phi + Gamma K + L C)^{-1} L\n');
F = Phi - Gam*K - L*C;
denD = poly(F);                       % det(zI - F)
% D(z) det(zI - F) is a first-degree polynomial; fit it from z = 0 and z = 1
vals = zeros(1, 2);
zz = [0 1];
for i = 1:2
    vals(i) = K*((zz(i)*eye(2) - F)\L)*polyval(denD, zz(i));
end
numD = [vals(2) - vals(1), vals(1)];
fprintf('%28s = [%.4f %.4f]\n', 'numerator of D', numD);
% expected: [0.7710 -0.2648]
fprintf('%28s = [%.4f %.4f %.4f]\n', 'denominator of D', denD);
% expected: [1 -0.0321 0.2445]
numG = [a, 1 - 2*a];
denG = poly([1, a]);
charPoly = conv(denG, denD) + [0, 0, conv(numG, numD)];   % pad to degree 4
fprintf('%28s = [%s]\n', '1 + D G = 0, polynomial', num2str(charPoly, '%.4f '));
% expected: [1 -1.4 0.94 -0.24 0.02]
rts = roots(charPoly);
fprintf('%28s =', 'roots');
for i = 1:4, fprintf(' %s', cstr(rts(i), 4)); end
fprintf('\n');
% expected: 0.2, 0.2, 0.5 +- 0.5j (any order)

%% 7. Discrete LQR, Q = C^T C, weight rho on m^2
% P <- Q + Phi' P Phi - Phi' P Gam (rho + Gam' P Gam)^-1 Gam' P Phi, from P = 0
% With the Control System Toolbox: [Kl, P] = dlqr(Phi, Gam, Q, rho)
fprintf('\n7. LQR, Q = C^T C\n');
Q = C'*C;
fig = figure('Color', 'w', 'Position', [100 100 1000 340]);
ax1 = subplot(1, 2, 1); hold on; box on; grid on;
ax2 = subplot(1, 2, 2); hold on; box on; grid on;
h1 = gobjects(0);  h2 = gobjects(0);  labs = {};
for rho = [10 1 0.1]
    P = zeros(2);
    for it = 1:200
        s = rho + Gam'*P*Gam;
        P = Q + Phi'*P*Phi - (Phi'*P*Gam)*(Gam'*P*Phi)/s;
    end
    Kl = (Gam'*P*Phi)/(rho + Gam'*P*Gam);
    pl = eig(Phi - Gam*Kl);
    fprintf('   rho = %4.1f:  K = [%.4f %.4f]   poles = %s, %s   |z| = %.4f, %.4f\n', ...
            rho, Kl(1), Kl(2), cstr(pl(1), 4), cstr(pl(2), 4), abs(pl(1)), abs(pl(2)));
    x = [1; 0];
    cs = zeros(1, 11);  ms = zeros(1, 11);
    for i = 1:11
        cs(i) = x(1);
        ms(i) = -Kl*x;
        x = (Phi - Gam*Kl)*x;
    end
    h1(end+1) = plot(ax1, 0:10, cs, 'o-', 'MarkerSize', 3);
    h2(end+1) = stairs(ax2, 0:10, ms);
    labs{end+1} = sprintf('rho = %g', rho);
end
% expected:
%   rho = 10.0:  K = [0.2756 0.2550]   poles = 0.7136, 0.3917                |z| = 0.7136, 0.3917
%   rho =  1.0:  K = [0.7007 0.5891]   poles = 0.3688 +- 0.2111j             |z| = 0.4250, 0.4250
%   rho =  0.1:  K = [1.4304 1.0608]   poles = 0.0855 +- 0.2607j             |z| = 0.2744, 0.2744
xlabel(ax1, 'k'); ylabel(ax1, 'c(k)'); title(ax1, 'LQR regulation from x(0) = [1, 0]');
xlabel(ax2, 't (s)'); ylabel(ax2, 'm(t)'); title(ax2, 'LQR control');
legend(ax1, h1, labs); legend(ax2, h2, labs);
print(fig, fullfile(figdir, 'ch14_lqr.png'), '-dpng', '-r150');

fprintf('\nFigures written to %s\n', figdir);

%% Local functions
function [Phi, Gam] = sample_plant(A, B, T)
% Exact ZOH sampling: expm([A B; 0 0] T) has Phi in its top-left block and
% Gamma in its top-right block.
[n, nu] = size(B);
M = zeros(n + nu);
M(1:n, 1:n) = A;
M(1:n, n+1:end) = B;
E = expm(M*T);
Phi = E(1:n, 1:n);
Gam = E(1:n, n+1:end);
end

function Mc = ctrb_matrix(Phi, Gam)
% Controllability matrix [Gam, Phi Gam, ..., Phi^(n-1) Gam].
n = size(Phi, 1);
Mc = zeros(n, n*size(Gam, 2));
for i = 0:n-1
    Mc(:, i+1) = Phi^i*Gam;
end
end

function Mo = obsv_matrix(Phi, C)
% Observability matrix [C; C Phi; ...; C Phi^(n-1)].
n = size(Phi, 1);
Mo = zeros(n, n);
for i = 0:n-1
    Mo(i+1, :) = C*Phi^i;
end
end

function K = ackermann(Phi, Gam, coef)
% State-feedback gain K (a row) with det(zI - Phi + Gam K) = alpha(z),
% coef = [1, a1, ..., an]:  K = [0 ... 0 1] Ctrb^-1 alpha(Phi).
n = size(Phi, 1);
lastRow = [zeros(1, n - 1), 1];
K = lastRow*(ctrb_matrix(Phi, Gam)\polyvalm(coef, Phi));
end

function L = observer_gain(Phi, C, coef)
% Prediction-observer gain L (a column) with det(zI - Phi + L C) = alpha(z).
% Duality: the transpose of Ackermann's gain for the pair (Phi', C').
L = ackermann(Phi', C', coef)';
end

function [t, c, m] = regulate(Phi, Gam, K, x0, steps, T)
% Run x(k+1) = (Phi - Gam K) x(k); also return c(t) between samples. Inside
% one period the input is constant, so the exact solution is
% c(kT + tau) = x1 + x2 (1 - e^{-tau}) + m (tau - 1 + e^{-tau}).
x = x0;
t = zeros(1, 20*steps + 1);  c = zeros(1, 20*steps + 1);  m = zeros(1, steps);
taus = linspace(0, T, 21);
taus = taus(1:20);
i = 0;
for k = 1:steps
    m(k) = -K*x;
    for tau = taus
        i = i + 1;
        t(i) = (k - 1)*T + tau;
        c(i) = x(1) + x(2)*(1 - exp(-tau)) + m(k)*(tau - 1 + exp(-tau));
    end
    x = Phi*x + Gam*m(k);
end
t(end) = steps*T;
c(end) = x(1);
end

function s = cstr(z, d)
% A complex number as text, e.g. 0.5000+0.5000j
s = sprintf(['%.' num2str(d) 'f%+.' num2str(d) 'fj'], real(z), imag(z));
end
