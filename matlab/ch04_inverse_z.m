%% Computers in the Loop, Chapter 4: Difference Equations and the Inverse z-Transform
% One convolution two ways, difference equations, inversion by partial
% fractions, long division and residues, a repeated pole, Parseval, and the
% practice exercise.
% Milad Siami
%
% Long division of E(z) = num(z)/den(z) is the impulse response of the
% filter with the same coefficients read in powers of z^-1 (numerator
% padded with leading zeros to the length of the denominator).
%
% Run this file from the matlab/ folder. Figures go to matlab/figures/.

clear; close all;
here = fileparts(mfilename('fullpath'));
figdir = fullfile(here, 'figures');
if ~exist(figdir, 'dir'), mkdir(figdir); end

N = 10;
k = 0:N-1;

%% One convolution two ways: unit step * 0.5^k = 2 - 0.5^k
fTime = conv(ones(1, N), 0.5.^k);
fTime = fTime(1:N);
fZ = long_division([1 0 0], conv([1 -1], [1 -0.5]), N);    % z^2/((z-1)(z-0.5))
fprintf('Convolution of the unit step with 0.5^k\n');
fprintf('  direct sum      :'); fprintf(' %.4f', fTime); fprintf('\n');
fprintf('  from E1(z)E2(z) :'); fprintf(' %.4f', fZ); fprintf('\n');
fprintf('  2 - 0.5^k       :'); fprintf(' %.4f', 2 - 0.5.^k); fprintf('\n');
% expected (all three): 1 1.5 1.75 1.875 1.9375 1.9688 1.9844 1.9922 1.9961 1.9980

%% Difference equation with a step input: m(k) = e(k) - e(k-1) - m(k-1)
e = ones(1, N);
m = zeros(1, N);
for i = 1:N                          % i = k + 1
    if i == 1
        m(i) = e(i);                 % e(-1) = m(-1) = 0
    else
        m(i) = e(i) - e(i-1) - m(i-1);
    end
end
fprintf('\nm(k) = e(k) - e(k-1) - m(k-1), unit step:'); fprintf(' %d', m);
fprintf('  (M(z) = z/(z+1): (-1)^k)\n');
% expected: 1 -1 1 -1 1 -1 1 -1 1 -1

%% x(k+2) - 3x(k+1) + 2x(k) = 0, x(0) = 0, x(1) = 1, inverted three ways
x = zeros(1, N);
x(2) = 1;
for i = 1:N-2
    x(i+2) = 3*x(i+1) - 2*x(i);
end
fprintf('\nx(k+2) - 3x(k+1) + 2x(k) = 0, x(0) = 0, x(1) = 1\n');
fprintf('  recursion       :'); fprintf(' %d', x); fprintf('\n');
% expected: 0 1 3 7 15 31 63 127 255 511

% (1) partial fractions of X(z)/z = 1/((z-1)(z-2))
[r, p] = residue(1, [1 -3 2]);
fprintf('  partial fractions of X(z)/z:');
for i = 1:numel(r)
    fprintf('  residue %g at z = %g', r(i), p(i));
end
fprintf('\n');
% expected: residue -1 at z = 1 and residue 1 at z = 2 (the order may differ)
pf = zeros(1, N);
for i = 1:numel(r)
    pf = pf + r(i)*p(i).^k;          % X(z) = sum r_i z/(z - p_i)
end
fprintf('  sum k_i p_i^k   :'); fprintf(' %d', round(real(pf))); fprintf('\n');
% expected: 0 1 3 7 15 31 63 127 255 511

% (2) long division of z/(z^2 - 3z + 2): filter([0 1 0], [1 -3 2], impulse)
fprintf('  long division   :'); fprintf(' %d', round(long_division([1 0], [1 -3 2], N))); fprintf('\n');
% expected: 0 1 3 7 15 31 63 127 255 511

% (3) residues of z^(k-1) X(z) = z^k/((z-1)(z-2)) at z = 2 and z = 1
res = 2.^k/(2 - 1) + 1/(1 - 2);
fprintf('  residues        :'); fprintf(' %d', round(res)); fprintf('  -> x(k) = 2^k - 1\n');
% expected: 0 1 3 7 15 31 63 127 255 511

%% A repeated pole: E(z) = z/((z-1)^2 (z-2))
den = conv(conv([1 -1], [1 -1]), [1 -2]);
[r, p] = residue(1, den);            % E(z)/z = 1/((z-1)^2 (z-2))
fprintf('\nRepeated pole, E(z) = z/((z-1)^2 (z-2))\n');
fprintf('  E(z)/z: residues'); fprintf(' %g', round(r*1e6)/1e6);
fprintf('  at poles'); fprintf(' %g', round(p*1e6)/1e6); fprintf('\n');
% expected: residue 1 at z = 2; at z = 1: -1 for 1/(z-1), then -1 for 1/(z-1)^2
fprintf('  (the repeated pole is listed twice: k11 for 1/(z-1), then k12 for 1/(z-1)^2)\n');
fprintf('  long division   :'); fprintf(' %d', round(long_division([1 0], den, N))); fprintf('\n');
fprintf('  2^k - k - 1     :'); fprintf(' %d', 2.^k - k - 1); fprintf('\n');
% expected (both): 0 0 1 4 11 26 57 120 247 502

%% No factor z in the numerator: E(z) = 1/(z - 0.5)
fprintf('\nE(z) = 1/(z - 0.5): long division'); fprintf(' %.4f', long_division(1, [1 -0.5], 6));
fprintf('  = -2 delta(k) + 2 (0.5)^k:'); fprintf(' %.4f', -2*(k(1:6) == 0) + 2*0.5.^k(1:6));
fprintf('\n');
% expected (both): 0 1 0.5 0.25 0.125 0.0625

%% Parseval: sum (0.5^k)^2 = 4/3, in time and in frequency
energyTime = sum(0.25.^(0:199));
energyFreq = integral(@(th) 1./(1.25 - cos(th)), 0, 2*pi)/(2*pi);
fprintf('\nParseval for 0.5^k: time %.6f, frequency %.6f, exact 4/3 = %.6f\n', ...
        energyTime, energyFreq, 4/3);
% expected: time 1.333333, frequency 1.333333, exact 4/3 = 1.333333

%% Practice exercise: E(z) = z^2/((z-1)(z-0.2))
fprintf('\nPractice exercise, E(z) = z^2/((z-1)(z-0.2))\n');
fprintf('  long division      :'); fprintf(' %.4f', long_division([1 0 0], conv([1 -1], [1 -0.2]), 6)); fprintf('\n');
fprintf('  5/4 - (1/4) 0.2^k  :'); fprintf(' %.4f', 1.25 - 0.25*0.2.^k(1:6)); fprintf('\n');
% expected (both): 1 1.2 1.24 1.248 1.2496 1.2499

% Check yourself: (3z^2 + z)/((z-1)(z+0.5))
fprintf('Check yourself, (3z^2 + z)/((z-1)(z+0.5)):');
fprintf(' %.4f', long_division([3 1 0], conv([1 -1], [1 0.5]), 5));
fprintf('  = 8/3 + (1/3)(-0.5)^k:'); fprintf(' %.4f', 8/3 + (-0.5).^k(1:5)/3); fprintf('\n');
% expected (both): 3 2.5 2.75 2.625 2.6875

%% Figure: the convolution and the Parseval integrand
th = linspace(0, 2*pi, 400);
fig = figure('Color', 'w', 'Position', [100 100 1000 340]);
subplot(1, 2, 1); hold on; box on;
stem(k, fTime);
plot([k(1) k(end)], [2 2], '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 0.8);
xlabel('k'); ylabel('f(k)'); title('Step * 0.5^k = 2 - 0.5^k');
subplot(1, 2, 2); hold on; box on;
plot(th, 1./(1.25 - cos(th)));
h = plot([0 2*pi], [4/3 4/3], '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 0.8);
xlabel('\theta (rad)'); ylabel('|F(e^{j\theta})|^2'); title('Parseval: same energy');
legend(h, {'mean = 4/3'}, 'FontSize', 8);
print(fig, fullfile(figdir, 'ch04_convolution_parseval.png'), '-dpng', '-r150');
fprintf('\nSaved figures/ch04_convolution_parseval.png\n');

%% Local function
function f = long_division(num, den, n)
% First n samples of E(z) = num/den (descending powers of z, deg num <= deg den):
% the impulse response of the filter with the same coefficients in z^-1.
num = [zeros(1, numel(den) - numel(num)), num];
imp = [1, zeros(1, n - 1)];
f = filter(num, den, imp);
end
