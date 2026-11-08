%% Computers in the Loop, Chapter 5: Realizations, State Variables, and the Sample-and-Hold
% G(z) = (z + 0.5)/(z^2 - 0.75z + 0.125) run as the direct form, the
% two-delay form and the state equations; the sampled-signal table; a ramp
% through the hold and the hold-error bound.
% Milad Siami
%
% Run this file from the matlab/ folder. Figures go to matlab/figures/.

clear; close all;
here = fileparts(mfilename('fullpath'));
figdir = fullfile(here, 'figures');
if ~exist(figdir, 'dir'), mkdir(figdir); end

b0 = 1;    b1 = 0.5;        % numerator    beta0 z + beta1
a1 = -0.75; a2 = 0.125;     % denominator  z^2 + alpha1 z + alpha2
N = 13;
e = ones(1, N);             % unit step from k = 0

%% Direct form: m_k = b0 e_{k-1} + b1 e_{k-2} - a1 m_{k-1} - a2 m_{k-2}
mDir = zeros(1, N);
for i = 1:N                 % i = k + 1; values before k = 0 are zero
    em1 = 0; em2 = 0; mm1 = 0; mm2 = 0;
    if i >= 2, em1 = e(i-1); mm1 = mDir(i-1); end
    if i >= 3, em2 = e(i-2); mm2 = mDir(i-2); end
    mDir(i) = b0*em1 + b1*em2 - a1*mm1 - a2*mm2;
end

%% Two-delay form: x2(k+1) = e_k - a1 x2 - a2 x1, x1(k+1) = x2, m = x2 + 0.5 x1
x1 = 0; x2 = 0;
mTwo = zeros(1, N); X1 = zeros(1, N); X2 = zeros(1, N);
for i = 1:N
    mTwo(i) = b0*x2 + b1*x1;
    X1(i) = x1;
    X2(i) = x2;
    x2new = e(i) - a1*x2 - a2*x1;
    x1 = x2;
    x2 = x2new;
end

%% State equations: x(k+1) = A x(k) + B e(k), m(k) = C x(k)
A = [0, 1; -a2, -a1];
B = [0; 1];
C = [b1, b0];
x = zeros(2, 1);
mSS = zeros(1, N);
for i = 1:N
    mSS(i) = C*x;
    x = A*x + B*e(i);
end

closedForm = 4*(1 - 0.5.^(0:N-1)).^2;
fprintf('G(z) = (z + 0.5)/(z^2 - 0.75z + 0.125), unit step from rest\n');
fprintf('  x1(k)            :'); fprintf(' %.4f', X1(1:6)); fprintf('\n');
% expected: 0 0 1 1.75 2.1875 2.4219
fprintf('  x2(k)            :'); fprintf(' %.4f', X2(1:6)); fprintf('\n');
% expected: 0 1 1.75 2.1875 2.4219 2.5430
fprintf('  direct form      :'); fprintf(' %.4f', mDir(1:6)); fprintf('\n');
fprintf('  two-delay form   :'); fprintf(' %.4f', mTwo(1:6)); fprintf('\n');
fprintf('  state equations  :'); fprintf(' %.4f', mSS(1:6)); fprintf('\n');
fprintf('  4 (1 - 0.5^k)^2  :'); fprintf(' %.4f', closedForm(1:6)); fprintf('\n');
% expected (all four): 0 1 2.25 3.0625 3.5156 3.7539
fprintf('  final value G(1) = %g\n', (1 + 0.5)/(1 - 0.75 + 0.125));
% expected: 4

% Partial fractions of M(z)/z = (z + 0.5)/((z^2 - 0.75z + 0.125)(z - 1))
[r, p] = residue([1 0.5], conv([1 a1 a2], [1 -1]));
fprintf('  residues of M(z)/z:');
for i = 1:numel(r)
    fprintf('  %.4f at z = %.4f', r(i), p(i));
end
fprintf('\n');
% expected: 4 at z = 0.25, -8 at z = 0.5, 4 at z = 1 (the order may differ)

% G(z) = C (zI - A)^-1 B at two test points
for z = [2.0, 1.1 + 0.9j]
    Gss = C*((z*eye(2) - A)\B);
    fprintf('  z = %s: C (zI - A)^-1 B = %s,  G(z) = %s\n', cstr(z, 1), cstr(Gss, 6), ...
            cstr((z + 0.5)/(z*z - 0.75*z + 0.125), 6));
end
% expected: z = 2: 0.952381 (both);  z = 1.1+0.9j: 0.387334-1.315096j (both)
fprintf('  det(zI - A) coefficients:'); fprintf(' %.4f', poly(A)); fprintf('\n');
% expected: 1 -0.75 0.125

%% The sampled-signal table: E(s) = 1/((s+1)(s+2)), T = ln 2
% E(z) = 0.25 z/((z - 0.5)(z - 0.25))
T = log(2);
k = 0:7;
samples = exp(-k*T) - exp(-2*k*T);
fromTable = filter([0 0.25 0], conv([1 -0.5], [1 -0.25]), [1, zeros(1, 7)]);
fprintf('\nE(s) = 1/((s+1)(s+2)), T = ln 2\n');
fprintf('  samples e(kT)                     :'); fprintf(' %.5f', samples); fprintf('\n');
fprintf('  series of 0.25z/((z-0.5)(z-0.25)) :'); fprintf(' %.5f', fromTable); fprintf('\n');
% expected (both): 0 0.25 0.1875 0.10938 0.05859 0.03027 0.01538 0.00775

%% A ramp through the zero-order hold, and the hold-error bound
T = 0.5;
t = linspace(0, 3, 3001);
held = T*floor(t/T + 1e-12);         % the held ramp
gap = t - held;
fprintf('\nRamp through the ZOH, T = %g: gap e(t) - ebar(t) between %.3f and %.3f, mean %.3f (about T/2 = %g)\n', ...
        T, min(gap), max(gap), mean(gap), T/2);
% expected: between 0.000 and 0.499, mean 0.249 (about T/2 = 0.25)
Tq = 0.05/(2*pi);                    % T max|de/dt| <= 0.05 for sin(2 pi t)
fprintf('Hold error for sin(2 pi t) below 0.05 needs T <= 0.05/(2 pi) = %.5f s (%.0f samples per cycle)\n', ...
        Tq, 1/Tq);
% expected: 0.00796 s (126 samples per cycle)
tt = linspace(0, 1, 200001);
err = abs(sin(2*pi*tt) - sin(2*pi*Tq*floor(tt/Tq)));
fprintf('  largest error with that T: %.4f\n', max(err));
% expected: 0.0500

%% Figure: step response of G(z), and same samples from different signals
fig = figure('Color', 'w', 'Position', [100 100 1000 340]);
subplot(1, 2, 1); hold on; box on;
stem(0:N-1, mDir);
plot([0 N-1], [4 4], '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 0.8);
xlabel('k'); ylabel('m_k'); title('Step response of G(z): every realization');
subplot(1, 2, 2); hold on; box on;
tt = linspace(0, 4, 800);
Ts = 0.5;
h1 = plot(tt, 1 - exp(-tt));
h2 = stairs(tt, 1 - exp(-Ts*floor(tt/Ts)));
h3 = plot(Ts*(0:8), 1 - exp(-Ts*(0:8)), 'o', 'MarkerSize', 4);
xlabel('t (s)'); title('Same samples, different signals');
legend([h1 h2 h3], {'e(t) = 1 - e^{-t}', 'held, T = 0.5 s', 'same samples'}, ...
       'FontSize', 8, 'Location', 'southeast');
print(fig, fullfile(figdir, 'ch05_realizations.png'), '-dpng', '-r150');
fprintf('\nSaved figures/ch05_realizations.png\n');

%% Local function
function s = cstr(z, d)
% A complex number as text, e.g. 0.387334-1.315096j
s = sprintf(['%.' num2str(d) 'f%+.' num2str(d) 'fj'], real(z), imag(z));
end
