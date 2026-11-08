%% Computers in the Loop, Chapter 3: Discrete-Time Systems and the z-Transform
% The satellite through a sample-and-hold, the z-transform series against
% closed forms, the shift and scaling rules, and the final value theorem on
% the position loop.
% Milad Siami
%
% Reminder on MATLAB's filter: the coefficient vectors are in powers of
% z^-1. For example z/(z^2 - 3z + 2) = z^-1/(1 - 3z^-1 + 2z^-2) is
% filter([0 1], [1 -3 2], x).
%
% Run this file from the matlab/ folder. Figures go to matlab/figures/.

clear; close all;
here = fileparts(mfilename('fullpath'));
figdir = fullfile(here, 'figures');
if ~exist(figdir, 'dir'), mkdir(figdir); end

%% The satellite: unit inertia, torque through a sample-and-hold
% Exact integration over each period with the torque held:
%   theta(k+1) = theta(k) + T omega(k) + T^2/2 tau(k),  omega(k+1) = omega(k) + T tau(k)
T = 1;
n = 8;
tau = ones(1, n + 2);            % constant torque from k = 0
theta = zeros(1, n + 2);
omega = zeros(1, n + 2);
for i = 1:n+1                    % i = k + 1
    theta(i+1) = theta(i) + T*omega(i) + T^2/2*tau(i);
    omega(i+1) = omega(i) + T*tau(i);
end
k = 0:n-1;
fprintf('Satellite, T = 1 s, unit torque from rest\n');
fprintf('  theta(k)          :'); fprintf(' %g', theta(1:n)); fprintf('\n');
% expected: 0 0.5 2 4.5 8 12.5 18 24.5
fprintf('  k^2/2             :'); fprintf(' %g', 0.5*k.^2); fprintf('\n');
% expected: 0 0.5 2 4.5 8 12.5 18 24.5

% theta(k+2) - 2 theta(k+1) + theta(k) = T^2/2 [tau(k+1) + tau(k)]
lhs = theta(3:n) - 2*theta(2:n-1) + theta(1:n-2);
rhs = T^2/2*(tau(2:n-1) + tau(1:n-2));
fprintf('  second-order difference equation holds: %d\n', max(abs(lhs - rhs)) < 1e-12);
% expected: 1 (true)

% The same samples from Theta/Tau = (T^2/2)(z + 1)/(z - 1)^2 driven by a step
num = [0, T^2/2, T^2/2];         % (T^2/2)(z^-1 + z^-2)
den = conv([1 -1], [1 -1]);      % 1 - 2z^-1 + z^-2
fprintf('  Theta/Tau = (T^2/2)(z+1)/(z-1)^2, step response:');
fprintf(' %g', filter(num, den, ones(1, n))); fprintf('\n');
% expected: 0 0.5 2 4.5 8 12.5 18 24.5
fprintf('  T = 0.5 s: T^2/2 = %g\n', 0.5^2/2);
% expected: 0.125

%% The definition E(z) = sum e_k z^-k and the region of convergence
fprintf('\nPartial sums of sum e_k z^-k (200 terms) against the closed form\n');
k = 0:199;
names  = {'a^k, a = 0.5', '(-0.8)^k', 'unit step'};
seqs   = {0.5.^k, (-0.8).^k, ones(1, 200)};
closed = {@(z) z./(z - 0.5), @(z) z./(z + 0.8), @(z) z./(z - 1)};
roc    = [0.5, 0.8, 1.0];
for i = 1:3
    for z = [1.5, 2.0*exp(0.7j)]
        s = sum(seqs{i} .* z.^(-k));
        fprintf('  %-12s z = %s: series %s  closed form %s   (ROC |z| > %g)\n', ...
                names{i}, cstr(z, 3), cstr(s, 6), cstr(closed{i}(z), 6), roc(i));
    end
end
% expected:
%   a^k, a = 0.5 z = 1.500+0.000j: series 1.500000+0.000000j
%   a^k, a = 0.5 z = 1.530+1.288j: series 1.189258-0.236817j
%   (-0.8)^k     z = 1.500+0.000j: series 0.652174+0.000000j
%   (-0.8)^k     z = 1.530+1.288j: series 0.737037+0.145432j
%   unit step    z = 1.500+0.000j: series 3.000000+0.000000j
%   unit step    z = 1.530+1.288j: series 1.272944-0.663926j
%   (each closed form equals its series)

z = 0.6;                         % inside the ROC of 0.5^k
fprintf('  0.5^k at z = 0.6 (inside the ROC): partial sums 50, 100, 200 terms:');
for N = [50 100 200]
    fprintf(' %.4f', sum(0.5.^k(1:N) .* z.^(-k(1:N))));
end
fprintf('   closed form %.4f\n', z/(z - 0.5));
% expected: 5.9993 6.0000 6.0000, closed form 6.0000
z = 0.4;                         % outside the ROC
fprintf('  0.5^k at z = 0.4 (outside the ROC): partial sums 10, 20, 40 terms:');
for N = [10 20 40]
    fprintf(' %.3g', sum(0.5.^k(1:N) .* z.^(-k(1:N))));
end
fprintf('  (diverge)\n');
% expected: 33.3 343 3.01e+04 (diverge)

%% Shift and scaling rules for e_k = 0.5^k
z = 1.7 + 0.4j;
E = @(z) z./(z - 0.5);
fprintf('\nShift rules for e_k = 0.5^k at z = %s\n', cstr(z, 1));
fprintf('  Z[e(k-3)] = z^-3 E(z)            = %s, closed form 1/(z^2(z-0.5)) = %s\n', ...
        cstr(z^-3*E(z), 6), cstr(1/(z^2*(z - 0.5)), 6));
% expected: 0.183553-0.183015j for both
fprintf('  Z[e(k+2)] = z^2 (E(z) - 1 - 0.5/z) = %s, 0.25 z/(z-0.5) = %s\n', ...
        cstr(z^2*(E(z) - 1 - 0.5/z), 6), cstr(0.25*z/(z - 0.5), 6));
% expected: 0.343750-0.031250j for both
fprintf('  Z[(-0.5)^k] by scaling, E(-z)    = %s, z/(z+0.5) = %s\n', ...
        cstr(E(-z), 6), cstr(z/(z + 0.5), 6));
% expected: 0.780000+0.040000j for both

%% The position loop and the final value theorem
% F(z) = (0.368z + 0.264)/((z-1)(z-0.368)), with 0.368 = e^-1, 0.264 = 1 - 2e^-1
e1 = exp(-1);
Fn = [e1, 1 - 2*e1];
Fd = conv([1 -1], [1 -e1]);
fprintf('\nPosition loop, F(z) = (0.368z + 0.264)/((z-1)(z-0.368)) (exact e^-1 used), unit step reference\n');
fig = figure('Color', 'w', 'Position', [100 100 640 340]);
hold on; box on;
h = gobjects(0);
for K = [1 3]
    charPoly = Fd + K*[0, Fn];          % (z-1)(z-0.368) + K(0.368z + 0.264)
    p = roots(charPoly);
    e = filter(Fd, charPoly, ones(1, 16));   % E/R = 1/(1 + K F)
    fprintf('  K = %d: characteristic polynomial [%s], poles %s, |z| = %.3f\n', ...
            K, num2str(charPoly, '%.3f  '), num2str(p.', 4), abs(p(1)));
    fprintf('         e(k) ='); fprintf(' %.3f', e(1:10)); fprintf('\n');
    h(end+1) = plot(0:15, e, 'o-', 'MarkerSize', 3);
end
% expected:
%   K = 1: [1 -1 0.632], poles 0.5 +- 0.618j, |z| = 0.795
%          e(k) = 1 0.632 0 -0.4 -0.4 -0.147 0.106 0.199 0.132 0.006
%   K = 3: [1 -0.264 1.161], poles 0.132 +- 1.069j, |z| = 1.077
%          e(k) = 1 -0.104 -1.188 -0.194 1.328 0.576 -1.389 -1.035 1.338 1.555
fprintf('  final value formula: lim_{z->1} z/(1 + K F(z)) = 0 for every K (F has a pole at z = 1);\n');
fprintf('  it is the limit of e(k) only when the loop is stable (0 < K < 2.39).\n');
plot([0 15], [0 0], 'k-', 'LineWidth', 0.5);
xlabel('k'); ylabel('e(k)'); title('Position loop: error samples');
legend(h, {'K = 1', 'K = 3'}, 'FontSize', 8);
print(fig, fullfile(figdir, 'ch03_position_error.png'), '-dpng', '-r150');
fprintf('\nSaved figures/ch03_position_error.png\n');

%% Local function
function s = cstr(z, d)
% A complex number as text, e.g. 1.189258-0.236817j
s = sprintf(['%.' num2str(d) 'f%+.' num2str(d) 'fj'], real(z), imag(z));
end
