%% Computers in the Loop, Chapter 11: Stability of Sampled-Data Systems
% The sampled integrator, Routh's array (epsilon rule, auxiliary polynomial),
% the bilinear substitution plus Routh, and Jury's test, on the chapter's
% loops; every gain range is cross-checked with the roots.
% Milad Siami
%
% Loops tested (characteristic polynomial Q(z), gain K):
%   ZOH + 1/(s(s+1)), T = 1 s, gain K:             0 < K < 2.39
%   the same loop with one sample of delay:        0 < K < 0.805
%   the laser loop with D = k, T = 0.1 s:          -2 < k < 54.7
%   practice exercise, accumulator on 0.5/(z-0.5): 0 < K < 6
%
% Run this file from the matlab/ folder. Figures go to matlab/figures/.

clear; close all;
here = fileparts(mfilename('fullpath'));
figdir = fullfile(here, 'figures');
if ~exist(figdir, 'dir'), mkdir(figdir); end

%% The sampled integrator under feedback: H(z) = T/(z - (1 - KT)), T = 0.1 s
T = 0.1;
fprintf('Sampled integrator T/(z-1) with feedback K, T = 0.1 s: H(z) = T/(z - (1 - KT))\n');
for K = [5 10 15 25]
    pole = 1 - K*T;
    hk = T*pole.^(0:199);               % impulse response h(k)
    s = sum(abs(hk));
    if abs(pole) >= 1, note = '  (grows)'; else, note = ''; end
    fprintf('  K = %2d: pole %5.2f,  sum |h| = %.3g%s\n', K, pole, s, note);
end
% expected: K = 5: 0.2;  K = 10: 0.1;  K = 15: 0.2;  K = 25: 3.31e+34 (grows)
fprintf('  stable for 0 < K < 2/T = %g\n', 2/T);
% expected: 20

%% Routh's array on the chapter's examples
fprintf('\nRouth: s^4 + s^3 + 2s^2 + 3s + 2\n');
[~, ch] = routh([1 1 2 3 2], true, 's');
fprintf('  sign changes: %d; roots %s\n', ch, num2str(roots([1 1 2 3 2]).', 3));
% expected: 2 sign changes (first column 1 1 -1 5 2); roots 0.33 +- 1.437j, -0.83 +- 0.482j

% Gain range of s^4 + 5s^3 + 8s^2 + 6s + K: no sign change in the first column
Kmax = edge_of(@(K) routh_changes([1 5 8 6 K]) == 0, 0.1, 12);
fprintf('Gain range of s^4 + 5s^3 + 8s^2 + 6s + K: 0 < K < %.4f (204/25 = %.4f)\n', Kmax, 204/25);
% expected: 8.1600 (204/25 = 8.1600)
fprintf('  at K = 204/25 the roots on the axis: %s\n', num2str(roots([1 5 8 6 204/25]).', 3));
% expected: -2.5 +- 0.742j and +- 1.095j

polys = {[1 0 -7 6], [1 3 4 12], [1 1 3 3 -4 -4]};
for i = 1:3
    c = polys{i};
    fprintf('Routh: [%s]\n', num2str(c));
    [~, ch] = routh(c, true, 's');
    fprintf('  right half-plane roots: %d;  roots %s\n', ch, num2str(roots(c).', 3));
end
% expected:
%   [1 0 -7 6]: zero first entry in row s^2 (eps rule); 2 right half-plane roots (-3, 2, 1)
%   [1 3 4 12]: row s^1 is zero, auxiliary 3s^2 + 12; 0 right half-plane roots (-3, +-2j)
%   [1 1 3 3 -4 -4]: row s^3 is zero, auxiliary s^4 + 3s^2 - 4; 1 right half-plane root
%                    (+-2j, 1, -1, -1)

%% The sampled loops
e1 = exp(-1);
% laser loop: exact ZOH equivalent of 2/((s+1)(s+4)) with T = 0.1 s
% With the Control System Toolbox: c2d(tf(2, [1 5 4]), 0.1, 'zoh')
[nl, dl] = zoh_tf(2, [1 5 4], 0.1);
nl = nl(end-1:end);
fprintf('\nLaser loop, T = 0.1 s: G(z) = (%.4f z + %.4f)/(z^2 - %.4f z + %.4f)\n', nl(1), nl(2), -dl(2), dl(3));
% expected: (0.0085 z + 0.0072)/(z^2 - 1.5752 z + 0.6065)

loopNames = {'ZOH + 1/(s(s+1)), T = 1 s, gain K', ...
             'same loop, one sample of delay (D = K/z)', ...
             'laser loop, T = 0.1 s, D = k', ...
             'practice exercise: D = Kz/(z-1), G = 0.5/(z-0.5)'};
Q = {@(K) [1, e1*K - (1 + e1), e1 + (1 - 2*e1)*K], ...
     @(K) [1, -(1 + e1), e1*(1 + K), (1 - 2*e1)*K], ...
     @(K) [1, dl(2) + nl(1)*K, dl(3) + nl(2)*K], ...
     @(K) [1, 0.5*K - 1.5, 0.5]};
lo = [0.01 0.01 0.01 0.01];
hi = [10 3 200 20];
fprintf('\nUpper gain limits, three ways\n');
for i = 1:4
    q = Q{i};
    kRoots = edge_of(@(K) all(abs(roots(q(K))) < 1), lo(i), hi(i));
    kBil = edge_of(@(K) bilinear_test(q(K)), lo(i), hi(i));
    kJury = edge_of(@(K) jury(q(K)), lo(i), hi(i));
    fprintf('  %-48s: roots %.4f, bilinear + Routh %.4f, Jury %.4f\n', loopNames{i}, kRoots, kBil, kJury);
end
% expected:
%   ZOH + 1/(s(s+1)), T = 1 s, gain K        : 2.3922 (all three)
%   same loop, one sample of delay (D = K/z) : 0.8047 (all three)
%   laser loop, T = 0.1 s, D = k             : 54.7128 (all three)
%   practice exercise                        : 6.0000 (all three)

qw = bilinear(Q{2}(0.5));
fprintf('\nThe delayed loop, K = 0.5: Q_w(w) ='); fprintf(' %.4f', qw); fprintf('\n');
% expected: 2.7876 4.2124 0.6839 0.3161
routh(qw, true, 'w');
% expected first column: 2.788 4.212 0.4748 0.3161 (no sign change: stable)
kLow = fzero(@(k) polyval(Q{3}(k), 1), [-10 0]);
fprintf('Laser loop lower limit: Q(1) = 0 at k = %.3f\n', kLow);
% expected: -2.000

%% Figure: largest pole size against K, and the stable triangle
Ks = linspace(0.01, 3, 600);
r0 = zeros(size(Ks));  r1 = zeros(size(Ks));
for i = 1:numel(Ks)
    r0(i) = max(abs(roots(Q{1}(Ks(i)))));
    r1(i) = max(abs(roots(Q{2}(Ks(i)))));
end
fig = figure('Color', 'w', 'Position', [100 100 1000 360]);
subplot(1, 2, 1); hold on; box on;
h1 = plot(Ks, r0);
h2 = plot(Ks, r1);
plot([Ks(1) Ks(end)], [1 1], 'k-', 'LineWidth', 0.6);
xlabel('K'); ylabel('largest |pole|'); title('ZOH + 1/(s(s+1)), T = 1 s');
legend([h1 h2], {'D = K', 'D = K/z (one sample of delay)'}, 'FontSize', 8, 'Location', 'northwest');
subplot(1, 2, 2); hold on; box on;
tri = [-2 1; 2 1; 0 -1; -2 1];
h1 = fill(tri(:,1), tri(:,2), [0.12 0.47 0.71], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
Kp = linspace(0, 3, 50);
h2 = plot(e1*Kp - (1 + e1), e1 + (1 - 2*e1)*Kp, 'r');
xlabel('a_1'); ylabel('a_0'); title('Second-order test for z^2 + a_1 z + a_0');
legend([h1 h2], {'stable: |a_0| < 1, Q(1) > 0, Q(-1) > 0', '(a_1, a_0) as K goes 0 to 3'}, ...
       'FontSize', 8, 'Location', 'southeast');
print(fig, fullfile(figdir, 'ch11_stability.png'), '-dpng', '-r150');
fprintf('\nSaved figures/ch11_stability.png\n');

%% Local functions
function [first, changes] = routh(c, verbose, var)
% Routh array of the polynomial with descending coefficients c. Returns the
% first column and the number of sign changes (roots in the right half-plane).
% A zero first entry is replaced by a small epsilon; a row of zeros is
% replaced by the derivative of the auxiliary polynomial.
epsval = 1e-6;
n = numel(c) - 1;
width = floor(n/2) + 1;
R = zeros(n + 1, width);              % row i holds the s^(n-i+1) row
r0 = c(1:2:end);  r1 = c(2:2:end);
R(1, 1:numel(r0)) = r0;
R(2, 1:numel(r1)) = r1;
notes = {};
for r = 1:n                           % R(r+1,:) is the row of s^(n-r)
    cur = R(r+1, :);
    if all(abs(cur) < 1e-12)          % a row of zeros
        up = R(r, :);
        d = n - r + 1;                % degree of the auxiliary polynomial
        jj = 0:width-1;
        aux = up(d - 2*jj >= 0);
        notes{end+1} = sprintf('row %s^%d is zero: auxiliary P(s) of degree %d, coefficients [%s]', ...
                               var, n - r, d, num2str(aux));
        keep = d - 2*jj > 0;
        newRow = up(keep).*(d - 2*jj(keep));   % coefficients of dP/ds
        cur = zeros(1, width);
        cur(1:numel(newRow)) = newRow;
        R(r+1, :) = cur;
    elseif abs(cur(1)) < 1e-12        % a zero first entry
        notes{end+1} = sprintf('zero first entry in row %s^%d: replaced by eps = %g', var, n - r, epsval);
        cur(1) = epsval;
        R(r+1, :) = cur;
    end
    if r == n, break; end
    up = R(r, :);
    nxt = zeros(1, width);
    for j = 1:width-1
        nxt(j) = (cur(1)*up(j+1) - up(1)*cur(j+1))/cur(1);
    end
    R(r+2, :) = nxt;
end
first = R(:, 1).';
changes = sum(sign(first(1:end-1)) ~= sign(first(2:end)));
if verbose
    for r = 1:n+1
        fprintf('    %s^%d: ', var, n - r + 1);
        fprintf('%9.4g  ', R(r, :));
        fprintf('\n');
    end
    for i = 1:numel(notes)
        fprintf('    %s\n', notes{i});
    end
end
end

function ch = routh_changes(c)
% Number of sign changes in the first column of the Routh array.
[~, ch] = routh(c, false, 's');
end

function qw = bilinear(c)
% Q_w(w) = (1-w)^n Q((1+w)/(1-w)) for Q with descending coefficients c.
n = numel(c) - 1;
qw = zeros(1, n + 1);
for i = 1:n+1
    k = n - i + 1;                    % c(i) multiplies z^k
    qw = qw + c(i)*conv(polypow([1 1], k), polypow([-1 1], n - k));
end
end

function p = polypow(base, k)
% base(w)^k as a coefficient vector.
p = 1;
for i = 1:k
    p = conv(p, base);
end
end

function ok = bilinear_test(c)
% Stable inside the unit circle: Q_w has no sign change and positive coefficients.
qw = bilinear(c);
ok = routh_changes(qw) == 0 && all(qw > 0);
end

function ok = jury(c)
% Jury's test on descending coefficients (leading coefficient > 0).
a = fliplr(c);                        % a0, a1, ..., an
n = numel(a) - 1;
ok = sum(a) > 0 && (-1)^n*polyval(c, -1) > 0 && abs(a(1)) < a(end);
if ~ok, return; end
row = a;
while numel(row) > 3
    m = numel(row) - 1;
    newRow = zeros(1, m);
    for j = 0:m-1
        newRow(j+1) = row(1)*row(j+1) - row(m+1)*row(m-j+1);
    end
    row = newRow;
    if ~(abs(row(1)) > abs(row(end)))
        ok = false;
        return
    end
end
end

function K = edge_of(test, lo, hi)
% Largest K in (lo, hi) at which test(K) still holds, by bisection.
% test(lo) must be true and test(hi) false.
for it = 1:100
    mid = (lo + hi)/2;
    if test(mid)
        lo = mid;
    else
        hi = mid;
    end
    if hi - lo < 1e-10, break; end
end
K = lo;
end

function [numz, denz] = zoh_tf(num, den, T)
% ZOH equivalent of num(s)/den(s) as polynomials in z (numerator padded
% to the length of the denominator).
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
