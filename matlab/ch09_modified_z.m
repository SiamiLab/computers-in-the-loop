%% Computers in the Loop, Chapter 9: The Modified z-Transform and Closed-Loop Systems
% Examples A1, A2, A3 of the modified z-transform, two samplers that are not
% synchronized, and the closed loops B1 and B3 checked by simulating the
% sampled loop with the plant integrated exactly.
% Milad Siami
%
% Convention (as in the chapter): F(z,m) = sum_k f((k+m)T) z^-(k+1), 0 <= m < 1,
% the transform of f delayed by (1-m)T.
%
% Run this file from the matlab/ folder. Figures go to matlab/figures/.

clear; close all;
here = fileparts(mfilename('fullpath'));
figdir = fullfile(here, 'figures');
if ~exist(figdir, 'dir'), mkdir(figdir); end

%% Example A1: f(t) = e^-t, F(z,m) = e^{-mT}/(z - e^{-T})
T = 0.7;
z = 1.5*exp(0.6j);
k = 0:399;
fprintf('Example A1: f(t) = e^-t, T = 0.7 s, at z = %s\n', cstr(z, 4));
% expected: z = 1.2380+0.8470j
for m = [0 0.3 0.8]
    s = sum(exp(-(k + m)*T) .* z.^(-(k + 1)));
    fprintf('  m = %g: series %s,  e^(-mT)/(z - e^-T) = %s\n', m, cstr(s, 6), ...
            cstr(exp(-m*T)/(z - exp(-T)), 6));
end
% expected: m = 0: 0.585154-0.668454j;  m = 0.3: 0.474316-0.541838j;
%           m = 0.8: 0.334245-0.381827j  (series and formula agree)
F = z/(z - exp(-T));
fprintf('  m = 0 gives F(z)/z = %s;  m -> 1 gives F(z) - f(0) = %s = e^-T/(z - e^-T) = %s\n', ...
        cstr(F/z, 6), cstr(F - 1, 6), cstr(exp(-T)/(z - exp(-T)), 6));
% expected: F(z)/z = 0.585154-0.668454j;  F(z) - f(0) = 0.290579-0.331944j (both)

%% Example A2: ZOH + 1/(s+1), unit step, T = 1 s
% C(z,m) = 1/(z-1) - e^(-mT)/(z - e^-T)
T = 1;
p = exp(-T);
fprintf('\nExample A2: ZOH + 1/(s+1), unit step, T = 1 s: C(z,m) = 1/(z-1) - e^(-mT)/(z - e^-T)\n');
for m = [0.25 0.5]
    num = [0, 1 - exp(-m*T), exp(-m*T) - p];
    c = zseries(num, conv([1 -1], [1 -p]), 6);
    c = c(2:end);                        % coefficients of z^-(k+1), k = 0, 1, ...
    fprintf('  m = %g: c((k+m)T) =', m); fprintf(' %.4f', c);
    fprintf(';  1 - e^-(k+m)T ='); fprintf(' %.4f', 1 - exp(-((0:4) + m)*T)); fprintf('\n');
end
% expected: m = 0.25: 0.2212 0.7135 0.8946 0.9612 0.9857 (both)
%           m = 0.5:  0.3935 0.7769 0.9179 0.9698 0.9889 (both)

%% Example A3: a delay of 0.4T (m = 0.6), T = 1 s
m = 0.6;
num = [1 - exp(-m*T), exp(-m*T) - p];
den = conv([1 0], [1 -p]);
fprintf('\nExample A3: delay 0.4T, m = %g: G(z,m) = (%.4f z + %.4f)/(z (z - %.4f))\n', m, num(1), num(2), p);
% expected: G(z,m) = (0.4512 z + 0.1809)/(z (z - 0.3679))
fprintf('  DC gain G(1,m) = %.4f;  zero at z = %.3f\n', polyval(num, 1)/polyval(den, 1), -num(2)/num(1));
% expected: DC gain 1.0000;  zero at z = -0.401
cs = filter([0, num], den, ones(1, 6));
fprintf('  step samples:'); fprintf(' %.4f', cs(1:4));
fprintf('   1 - e^-(k - 0.4):'); fprintf(' %.4f', [0, 1 - exp(-((1:3) - 0.4))]); fprintf('\n');
% expected (both): 0 0.4512 0.7981 0.9257

%% Two samplers that are not synchronized
T = 0.05;  h = 0.2;
p = exp(-T);
g1 = [1 - exp(-h*T), exp(-h*T) - p];      % G1(z,h) numerator over z(z - p)
mm = 1 - h;
g2 = [mm*T, (1 - mm)*T];                  % G2(z,0.8) numerator over z(z - 1)
fprintf('\nTwo samplers, T = %g s, offset hT = %.2f s\n', T, h*T);
fprintf('  G1(z,0.2) = (%.5f z + %.5f)/(z (z - %.5f))\n', g1(1), g1(2), p);
% expected: (0.00995 z + 0.03882)/(z (z - 0.95123))
fprintf('  G2(z,0.8) = (%.2f z + %.2f)/(z (z - 1))\n', g2(1), g2(2));
% expected: (0.04 z + 0.01)/(z (z - 1))
fprintf('  c(T) = %.6f  (a(0.01) = %.5f held for 0.04 s: %.6f)\n', g2(1)*g1(1), ...
        1 - exp(-0.01), 0.04*(1 - exp(-0.01)));
% expected: c(T) = 0.000398  (a(0.01) = 0.00995 held for 0.04 s: 0.000398)

%% Example B1: ZOH + 1/(s+1) in a unity loop, T = ln 2, simulated
% e(k) = r - c(k); the plant is driven by the held error.
T = log(2);
[Phi, Gam] = step_matrices(-1, 1, T);
x = 0;
cB1 = zeros(1, 8);
for i = 1:8
    cB1(i) = x;
    x = Phi*x + Gam*(1 - x);
end
fprintf('\nExample B1, T = ln 2: simulated c(kT) ='); fprintf(' %.4f', cB1);
fprintf('   (C/R = 0.5/z: 0, 0.5, 0.5, ...)\n');
% expected: 0 0.5 0.5 0.5 0.5 0.5 0.5 0.5

%% Example B3: two samplers, C/R = G1G2/(1 + G1G2 + HG2(z))
% E1 = R - C, E2 = G1 E1* - H C, C = G2 E2*  (H acts on the continuous C)
% States: x1 (plant of G1), xc (plant of G2, the output), xh (H = 1/(s+3)).
T = 0.3;
A = [-1 0 0; 0 -2 0; 0 1 -3];
B = [1 0; 0 1; 0 0];
[Phi, Gam] = step_matrices(A, B, T);
x = zeros(3, 1);
cSim = zeros(1, 25);
for i = 1:25
    cSim(i) = x(2);
    e1 = 1 - x(2);
    e2 = x(1) - x(3);
    x = Phi*x + Gam*[e1; e2];
end
% The same loop from the sampled-graph formula, everything over one denominator.
% With the Control System Toolbox: c2d(tf(1, [1 1]), T, 'zoh'), and so on.
[n1, d1] = zoh_tf(1, [1 1], T);
[n2, d2] = zoh_tf(1, [1 2], T);
[nh, dh] = zoh_tf(1, conv([1 2], [1 3]), T);       % HG2(z) = Z[ZOH H(s)/(s+2)]
N12 = conv(n1, n2);
D12 = conv(d1, d2);
num = conv(N12, dh);
den = conv(D12, dh) + conv(N12, dh) + conv(nh, D12);
num = [zeros(1, numel(den) - numel(num)), num];    % align the powers of z
cZ = filter(num, den, ones(1, 25));
fprintf('Example B3 (T = 0.3 s), unit step:\n');
fprintf('  simulated loop      :'); fprintf(' %.5f', cSim(1:8)); fprintf('\n');
fprintf('  G1G2/(1+G1G2+HG2(z)):'); fprintf(' %.5f', cZ(1:8)); fprintf('\n');
% expected (both): 0 0 0.05847 0.13225 0.19772 0.24601 0.27692 0.29410
fprintf('  largest difference over 25 samples: %.1e\n', max(abs(cSim - cZ)));
% expected: about 1e-15 (round-off only)

%% Figure: the output between the samples, and the two-sampler loop
% Left: Example A2, sweeping m traces the output between the samples.
T = 1;  p = exp(-1);
tt = [];  cc = [];
for m = linspace(0, 0.99, 34)
    num = [0, 1 - exp(-m*T), exp(-m*T) - p];
    c = zseries(num, conv([1 -1], [1 -p]), 8);      % c(k+2) = c((k+m)T)
    for kk = 0:5
        tt(end+1) = (kk + m)*T;
        cc(end+1) = c(kk + 2);
    end
end
tc = linspace(0, 6, 600);
fig = figure('Color', 'w', 'Position', [100 100 1000 340]);
subplot(1, 2, 1); hold on; box on;
h1 = plot(tc, 1 - exp(-tc), 'k', 'LineWidth', 1);
h2 = plot(tt, cc, '.', 'MarkerSize', 6);
h3 = plot(0:6, 1 - exp(-(0:6)), 'o', 'Color', [0.84 0.15 0.16]);
xlabel('t (s)'); title('Example A2: between the samples');
legend([h1 h2 h3], {'1 - e^{-t}', 'C(z,m), m swept', 'samples'}, 'FontSize', 8, 'Location', 'southeast');
% Right: Example B3, simulated loop against the sampled-graph formula
subplot(1, 2, 2); hold on; box on;
h1 = plot(0.3*(0:24), cSim, 'o', 'MarkerSize', 4);
h2 = plot(0.3*(0:24), cZ, 'x', 'MarkerSize', 5);
xlabel('t (s)'); title('Example B3 with two samplers, T = 0.3 s');
legend([h1 h2], {'simulated loop', 'G1G2/(1 + G1G2 + HG2)'}, 'FontSize', 8, 'Location', 'southeast');
print(fig, fullfile(figdir, 'ch09_modified_z.png'), '-dpng', '-r150');
fprintf('\nSaved figures/ch09_modified_z.png\n');

%% Local functions
function c = zseries(num, den, n)
% First n coefficients of num(z)/den(z) in powers of z^-1.
num = [zeros(1, numel(den) - numel(num)), num];
c = filter(num, den, [1, zeros(1, n - 1)]);
end

function [Phi, Gam] = step_matrices(A, B, T)
% Phi = e^{AT}, Gam = int_0^T e^{A s} ds B, from one matrix exponential.
nx = size(A, 1);
nu = size(B, 2);
M = zeros(nx + nu);
M(1:nx, 1:nx) = A;
M(1:nx, nx+1:end) = B;
E = expm(M*T);
Phi = E(1:nx, 1:nx);
Gam = E(1:nx, nx+1:end);
end

function [numz, denz] = zoh_tf(num, den, T)
% ZOH equivalent of num(s)/den(s) as polynomials in z. The numerator keeps
% its leading zero, so it has the same length as the denominator.
[A, B, C, D] = tf2ss_ccf(num, den);
[Phi, Gam] = step_matrices(A, B, T);
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

function s = cstr(z, d)
% A complex number as text, e.g. 0.585154-0.668454j
s = sprintf(['%.' num2str(d) 'f%+.' num2str(d) 'fj'], real(z), imag(z));
end
