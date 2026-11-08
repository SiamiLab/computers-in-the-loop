%% Computers in the Loop, Chapter 16: How Fast Must Networked Agents Sample? Denser Networks Require Faster Agents
% Agreement by hand, the largest stable update period of one agent (one
% consensus mode) and of path, ring, star and complete graphs, fast and slow
% agents, links added one at a time, and the steady-state disagreement under
% noise (ring against complete graph).
% Milad Siami
%
% Model (forward Euler, as in the notes): x(k+1) = x + T v, v(k+1) = v + T m,
% with m = -Lx x - Lv v. Each eigenvalue lambda of the Laplacian gives one
% mode with alpha = beta = lambda when Lx = Lv = L.
% Figures (in matlab/figures/):
%   ch16_agreement.png     three agents in a line agree by hand (T = 0.5) or not (T = 0.8)
%   ch16_roots.png         Euler roots of one agent as T grows (alpha = beta = 1 and 5)
%   ch16_tmax_vs_lmax.png  largest stable T against the largest Laplacian eigenvalue
%   ch16_fast_slow.png     positions of five agents: fast and slow agents, sparse and dense graphs
%   ch16_density.png       largest stable T as links are added, path to complete, 8 agents
%   ch16_crossing.png      steady-state disagreement under noise: ring against complete graph
%
% Change the CAPITAL constants to experiment. Run this file from the matlab/ folder.

clear; close all;
here = fileparts(mfilename('fullpath'));
figdir = fullfile(here, 'figures');
if ~exist(figdir, 'dir'), mkdir(figdir); end

SPARSE = [29 95 168]/255;             % the notes' blue
DENSE  = [200 16 46]/255;             % the notes' red
AGENTS = [29 95 168; 200 16 46; 46 125 50; 230 126 34; 31 42 51]/255;   % agents 1-5

%% 0. Agreement by hand: agents that set their own velocity, x(k+1) = x(k) - T L x(k)
fprintf('Two agents, one link, from x = (4, 0):\n');
L2 = laplacian([1 2], 2);
for T = [0.25 1.0 1.25]
    x = [4; 0];
    fprintf('  T = %g: (%g, %g)', T, x(1), x(2));
    for k = 1:3
        x = x - T*L2*x;
        fprintf('  (%g, %g)', x(1), x(2));
    end
    fprintf('   gap factor 1 - 2T = %g\n', 1 - 2*T);
end
% expected:
%   T = 0.25: (4, 0)  (3, 1)  (2.5, 1.5)  (2.25, 1.75)   gap factor 1 - 2T = 0.5
%   T = 1:    (4, 0)  (0, 4)  (4, 0)  (0, 4)              gap factor 1 - 2T = -1
%   T = 1.25: (4, 0)  (-1, 5)  (6.5, -2.5)  (-4.75, 8.75) gap factor 1 - 2T = -1.5
L3 = laplacian([1 2; 2 3], 3);
ev3 = round(eig(L3)*1e6)/1e6;
ev3(ev3 == 0) = 0;                    % print 0, not -0
fprintf('Three agents in a line: eigenvalues of L ='); fprintf(' %g', ev3);
fprintf('; agreement for T < 2/3\n');
% expected: 0 1 3

fig = figure('Color', 'w', 'Position', [100 100 900 320]);
Tlist = [0.5 0.8];
for j = 1:2
    T = Tlist(j);
    x = [3; 0; 0];
    X = zeros(11, 3);
    X(1, :) = x';
    for k = 1:10
        x = x - T*L3*x;
        X(k+1, :) = x';
    end
    subplot(1, 2, j); hold on; box on;
    h = gobjects(1, 3);
    for i = 1:3
        h(i) = plot(0:10, X(:, i), '-o', 'MarkerSize', 3, 'Color', AGENTS(i, :));
    end
    title(sprintf('three agents in a line, T = %g', T)); xlabel('k');
    if T > 0.6, ylim([-12 12]); else, ylim([-1 3.4]); end
    if j == 1
        legend(h, {'agent 1', 'agent 2', 'agent 3'}, 'FontSize', 8);
    end
end
print(fig, fullfile(figdir, 'ch16_agreement.png'), '-dpng', '-r150');
fprintf('  saved figures/ch16_agreement.png\n');
fprintf('Two masses with a spring and a damper (alpha = beta = 2): Tmax = %g\n', tmax_euler(2, 2));
% expected: 1

%% 1. One agent, m = -alpha x - beta v: the bound and where the roots go
% Euler model:  T < beta/alpha if beta^2 < 4 alpha (complex pair leaves through |z| = 1),
%               otherwise T < 4/(beta + sqrt(beta^2 - 4 alpha)) (a real root leaves through z = -1).
% Exact ZOH model (Gamma = [T^2/2; T]):  T < min(2/beta, 2 beta/alpha).
fprintf('One agent, m = -alpha x - beta v\n');
ab = [1 1; 5 5; 2 1; 2 4];
for i = 1:4
    fprintf('  alpha = %d, beta = %d:  Tmax Euler = %.4f,  exact model = %.4f\n', ...
            ab(i,1), ab(i,2), tmax_euler(ab(i,1), ab(i,2)), tmax_exact(ab(i,1), ab(i,2)));
end
% expected:
%   alpha = 1, beta = 1:  Tmax Euler = 1.0000,  exact model = 2.0000
%   alpha = 5, beta = 5:  Tmax Euler = 0.5528,  exact model = 0.4000
%   alpha = 2, beta = 1:  Tmax Euler = 0.5000,  exact model = 1.0000
%   alpha = 2, beta = 4:  Tmax Euler = 0.5858,  exact model = 0.5000

fig = figure('Color', 'w', 'Position', [100 100 520 460]);
hold on; box on;
th = linspace(0, 2*pi, 300);
plot(cos(th), sin(th), 'Color', [0.2 0.2 0.2], 'LineWidth', 1);
cases = {[1 1], SPARSE, 1.0; [5 5], DENSE, tmax_euler(5, 5)};
for i = 1:2
    a = cases{i, 1}(1);  b = cases{i, 1}(2);
    s = roots([1 b a]);               % the Euler roots move as z = 1 + T s
    for j = 1:numel(s)
        Tline = linspace(0, cases{i, 3}, 50);
        z = 1 + Tline*s(j);
        plot(real(z), imag(z), 'Color', cases{i, 2}, 'LineWidth', 2.5);
        plot(real(z(end)), imag(z(end)), 'o', 'Color', cases{i, 2}, 'MarkerFaceColor', cases{i, 2});
    end
end
axis equal;
xl = xlim;  yl = ylim;
plot(xl, [0 0], 'Color', [0.6 0.6 0.6], 'LineWidth', 0.6);
plot([0 0], yl, 'Color', [0.6 0.6 0.6], 'LineWidth', 0.6);
title({'Euler roots z = 1 + Ts: blue \alpha = \beta = 1 (to T = 1),', ...
       'red \alpha = \beta = 5 (to T = 0.553)'}, 'FontSize', 9);
print(fig, fullfile(figdir, 'ch16_roots.png'), '-dpng', '-r150');
fprintf('  saved figures/ch16_roots.png\n');

%% 2. Five agents, four graphs, Lx = Lv = L
N_AGENTS = 5;
G5 = four_graphs(N_AGENTS);
gnames = {'path', 'ring', 'star', 'complete'};
fprintf('\n%d agents, Lx = Lv = L\n', N_AGENTS);
fprintf('  graph      links  dmax  lambda_max  Tmax(formula)  Tmax(bisection)  exact model  platoon\n');
EIG = struct();
for i = 1:4
    name = gnames{i};
    edges = G5.(name);
    L = laplacian(edges, N_AGENTS);
    lam = sort(eig(L));
    EIG.(name) = lam;
    tForm = inf;  tExact = inf;  tPlat = inf;
    for j = 1:numel(lam)
        if j > 1                          % skip the zero eigenvalue (common motion)
            tForm = min(tForm, tmax_euler(lam(j), lam(j)));
            tExact = min(tExact, tmax_exact(lam(j), lam(j)));
        end
        tPlat = min(tPlat, tmax_euler(lam(j) + 1, lam(j) + 1));   % platoon: every mode shifted by 1
    end
    tBis = largest_stable_T(L, L, 3.0);
    fprintf('  %-9s  %5d  %4d  %10.4f  %13.4f  %15.4f  %11.4f  %7.4f\n', name, size(edges, 1), ...
            round(max(diag(L))), lam(end), tForm, tBis, tExact, tPlat);
    lamPrint = round(lam*1e4)/1e4;
    lamPrint(lamPrint == 0) = 0;
    fprintf('             eigenvalues'); fprintf(' %.4f', lamPrint); fprintf('\n');
end
% expected:
%   path           4     2      3.6180         1.0000           1.0000       0.5528   0.6342
%              eigenvalues 0 0.382 1.382 2.618 3.618
%   ring           5     2      3.6180         1.0000           1.0000       0.5528   0.6342
%              eigenvalues 0 1.382 1.382 3.618 3.618
%   star           4     4      5.0000         0.5528           0.5528       0.4000   0.4226
%              eigenvalues 0 1 1 1 5
%   complete      10     4      5.0000         0.5528           0.5528       0.4000   0.4226
%              eigenvalues 0 5 5 5 5

lm = linspace(0.05, 10, 400);
tE = zeros(size(lm));  tX = zeros(size(lm));
for i = 1:numel(lm)
    tE(i) = tmax_euler(lm(i), lm(i));
    tX(i) = tmax_exact(lm(i), lm(i));
end
fig = figure('Color', 'w', 'Position', [100 100 560 380]);
hold on; box on;
h1 = plot(lm, tE, 'Color', DENSE, 'LineWidth', 2.5);
h2 = plot(lm, tX, '--', 'Color', SPARSE, 'LineWidth', 1.6);
mk = {'o', 's'};
gm = {'path', 'star'};
for i = 1:2
    l = EIG.(gm{i})(end);
    plot(l, tmax_euler(l, l), mk{i}, 'Color', DENSE, 'MarkerFaceColor', DENSE, 'MarkerSize', 7);
    plot(l, tmax_exact(l, l), mk{i}, 'Color', SPARSE, 'MarkerSize', 7);
end
ylim([0 1.25]);
xlabel('largest Laplacian eigenvalue'); ylabel('largest stable T');
legend([h1 h2], {'Euler model', 'exact model'});
title('Lx = Lv = L; circles: path and ring, squares: star and complete', 'FontSize', 9);
print(fig, fullfile(figdir, 'ch16_tmax_vs_lmax.png'), '-dpng', '-r150');
fprintf('  saved figures/ch16_tmax_vs_lmax.png\n');

%% 3. Simulations: fast and slow agents, sparse and dense graphs
X0 = [2; -1; 1.5; -2.5; 0];
V0 = [0; 0.5; -0.5; 0.3; -0.3];
T_END = 12;
simCases = {'ring', 0.25, 'ring, T = 0.25 s (fast agents)', 6; ...
            'ring', 1.05, 'ring, T = 1.05 s (slow agents)', 6; ...
            'path', 0.6, 'path (sparse), T = 0.6 s', 8; ...
            'complete', 0.6, 'complete (dense), T = 0.6 s', 8};
fig = figure('Color', 'w', 'Position', [100 100 900 600]);
for c = 1:4
    T = simCases{c, 2};
    L = laplacian(G5.(simCases{c, 1}), N_AGENTS);
    F = network(L, L, T);
    nSteps = floor(T_END/T + 1e-9);
    s = [X0; V0];
    X = zeros(nSteps + 1, N_AGENTS);
    X(1, :) = s(1:N_AGENTS)';
    for k = 1:nSteps
        s = F*s;
        X(k+1, :) = s(1:N_AGENTS)';
    end
    t = (0:nSteps)*T;
    subplot(2, 2, c); hold on; box on; grid on;
    for i = 1:N_AGENTS
        plot(t, X(:, i), '-o', 'MarkerSize', 2.5, 'LineWidth', 1, 'Color', AGENTS(i, :));
    end
    ylim([-simCases{c, 4} simCases{c, 4}]);
    title(simCases{c, 3}, 'FontSize', 10);
    if c >= 3, xlabel('t = kT (s)'); end
    if mod(c, 2) == 1, ylabel('x_i(kT)'); end
end
print(fig, fullfile(figdir, 'ch16_fast_slow.png'), '-dpng', '-r150');
fprintf('  saved figures/ch16_fast_slow.png\n');

%% 4. Adding links one at a time, 8 agents (path, then ring, then chords, to complete)
N8 = 8;
edges = [(1:N8-1)', (2:N8)'];         % the path
order = [1 N8];                       % close the ring
for k = [2 3]                         % chords of length 2, then 3
    for i = 0:N8-1
        order(end+1, :) = sort([i, mod(i + k, N8)]) + 1;
    end
end
for i = 0:N8/2-1                      % the diameters
    order(end+1, :) = [i, i + 4] + 1;
end
seq = {edges};
for r = 1:size(order, 1)
    if ~ismember(order(r, :), edges, 'rows')
        edges(end+1, :) = order(r, :);
        seq{end+1} = edges;
    end
end
g = @(l) tmax_euler(l, l);            % Lx = Lv = L
nG = numel(seq);
links = zeros(1, nG);  tF = zeros(1, nG);  tB = zeros(1, nG);
loB = zeros(1, nG);  upB = zeros(1, nG);
for i = 1:nG
    L = laplacian(seq{i}, N8);
    lmax = max(eig(L));
    d = max(diag(L));
    links(i) = size(seq{i}, 1);
    tF(i) = g(lmax);
    tB(i) = largest_stable_T(L, L, 3.0);
    loB(i) = g(2*d);                   % degree band: g(2 dmax) <= Tmax <= g(dmax + 1)
    upB(i) = g(d + 1);
end
fprintf('\n%d agents, links added one at a time: largest stable T (formula, bisection)\n', N8);
for i = 1:nG
    fprintf('  %2d links: %.4f  %.4f\n', links(i), tF(i), tB(i));
end
% expected (formula and bisection agree):
%    7 links: 1.0000   8 links: 1.0000   9 links: 0.6723  10 links: 0.5889
%   11 links: 0.5141  12 links: 0.4701  13 links: 0.4701  14 links: 0.4503
%   15 links: 0.4226  16 links: 0.4226  17 links: 0.3497  18 links: 0.3411
%   19 links: 0.3253  20 links: 0.3164  21 links: 0.2998  22 to 28 links: 0.2929
fig = figure('Color', 'w', 'Position', [100 100 640 380]);
hold on; box on;
[xx, yLo] = mid_steps(links, loB);
[~, yUp] = mid_steps(links, upB);
h1 = fill([xx, fliplr(xx)], [yLo, fliplr(yUp)], SPARSE, 'FaceAlpha', 0.15, 'EdgeColor', 'none');
[xx, yy] = mid_steps(links, tF);
plot(xx, yy, 'Color', DENSE, 'LineWidth', 1.5);
h2 = plot(links, tB, 'o', 'Color', DENSE, 'MarkerFaceColor', DENSE, 'MarkerSize', 4);
xlabel('number of links (path = 7, ring = 8, complete = 28)'); ylabel('largest stable T');
ylim([0 1.12]);
legend([h1 h2], {'degree band g(2 d_{max}) .. g(d_{max} + 1)', 'largest stable T'}, 'FontSize', 8);
print(fig, fullfile(figdir, 'ch16_density.png'), '-dpng', '-r150');
fprintf('  saved figures/ch16_density.png\n');

%% 5. Noise: steady-state disagreement, ring against complete graph
% Per mode (alpha = beta = lambda), unit-intensity noise on the velocities:
%   p11 = (2 - T beta + T^2 alpha)/(alpha D),  p22 = 2/D,
%   D = (beta - T alpha)(4 - 2 T beta + T^2 alpha).
% Vx and Vv sum p11 and p22 over the modes 2..n.
fprintf('\nSteady-state disagreement (closed form; Lyapunov equation of the network)\n');
fprintf('   T     ring Vv   complete Vv   ring Vx   complete Vx\n');
Lring = laplacian(G5.ring, N_AGENTS);
Lcomp = laplacian(G5.complete, N_AGENTS);
for T = [0.1 0.2 0.4 0.5]
    fprintf('  %.1f  %8.3f  %11.3f  %8.3f  %11.3f   (check: %.3f, %.3f)\n', T, ...
            Vsum(EIG.ring, T, 'v'), Vsum(EIG.complete, T, 'v'), ...
            Vsum(EIG.ring, T, 'x'), Vsum(EIG.complete, T, 'x'), ...
            V_lyapunov(Lring, T, 'v'), V_lyapunov(Lcomp, T, 'v'));
end
% expected:
%   0.1     1.231        0.583     0.670        0.090   (check: 1.231, 0.583)
%   0.2     1.545        0.909     0.765        0.109   (check: 1.545, 0.909)
%   0.4     2.642        3.333     1.106        0.267   (check: 2.642, 3.333)
%   0.5     3.672       12.800     1.430        0.960   (check: 3.672, 12.800)
crossV = fzero(@(T) Vsum(EIG.ring, T, 'v') - Vsum(EIG.complete, T, 'v'), [0.2 0.55]);
crossX = fzero(@(T) Vsum(EIG.ring, T, 'x') - Vsum(EIG.complete, T, 'x'), [0.3 0.5527]);
fprintf('  crossing: T = %.4f (velocities), T = %.4f (positions); complete graph unstable above %.4f\n', ...
        crossV, crossX, 1 - 1/sqrt(5));
% expected: T = 0.3566 (velocities), T = 0.5172 (positions); unstable above 0.5528

fig = figure('Color', 'w', 'Position', [100 100 900 380]);
kind = {'v', 'x'};
labs = {'velocity disagreement V_v', 'position disagreement V_x'};
crossT = [crossV, crossX];
Tr = linspace(0.001, 0.75, 400);
Tc = linspace(0.001, 0.548, 400);
for j = 1:2
    vr = zeros(size(Tr));  vc = zeros(size(Tc));
    for i = 1:numel(Tr)
        vr(i) = Vsum(EIG.ring, Tr(i), kind{j});
        vc(i) = Vsum(EIG.complete, Tc(i), kind{j});
    end
    subplot(1, 2, j);
    h1 = semilogy(Tr, vr, 'Color', SPARSE, 'LineWidth', 2.2); hold on; box on; grid on;
    h2 = semilogy(Tc, vc, 'Color', DENSE, 'LineWidth', 2.2);
    semilogy([1 1]*(1 - 1/sqrt(5)), [0.05 60], '--', 'Color', DENSE, 'LineWidth', 1);
    semilogy(crossT(j), Vsum(EIG.ring, crossT(j), kind{j}), 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 5);
    xlabel('T (s)'); title(labs{j}, 'FontSize', 10);
    ylim([0.05 60]);
    if j == 1
        legend([h1 h2], {'ring (sparse)', 'complete (dense)'}, 'FontSize', 8, 'Location', 'northwest');
    end
end
print(fig, fullfile(figdir, 'ch16_crossing.png'), '-dpng', '-r150');
fprintf('  saved figures/ch16_crossing.png\n');

%% Local functions
function L = laplacian(edges, n)
% Graph Laplacian of an undirected graph; edges is a k x 2 list of node numbers.
L = zeros(n);
for r = 1:size(edges, 1)
    i = edges(r, 1);  j = edges(r, 2);
    L(i, i) = L(i, i) + 1;
    L(j, j) = L(j, j) + 1;
    L(i, j) = L(i, j) - 1;
    L(j, i) = L(j, i) - 1;
end
end

function G = four_graphs(n)
% Edge lists of the path, ring, star and complete graphs on n nodes.
G.path = [(1:n-1)', (2:n)'];
G.ring = [(1:n)', [2:n, 1]'];
G.star = [ones(n-1, 1), (2:n)'];
G.complete = nchoosek(1:n, 2);
end

function T = tmax_euler(alpha, beta)
% Largest stable T of x(k+1) = x + T v, v(k+1) = v - T(alpha x + beta v).
if beta^2 < 4*alpha
    T = beta/alpha;                           % complex pair leaves through |z| = 1
else
    T = 4/(beta + sqrt(beta^2 - 4*alpha));    % real root leaves through z = -1
end
end

function T = tmax_exact(alpha, beta)
% Same feedback, exact zero-order-hold model (Gamma = [T^2/2; T]).
T = min(2/beta, 2*beta/alpha);
end

function F = network(Lx, Lv, T)
% State matrix of the whole network, state [x; v].
n = size(Lx, 1);
F = [eye(n), T*eye(n); -T*Lx, eye(n) - T*Lv];
end

function r = disagreement_radius(Lx, Lv, T)
% Largest root size of the network once the common motion (mode 1) is removed.
[U, D] = eig(Lx);
[~, idx] = sort(diag(D));
U = U(:, idx);
W = U(:, 2:end);                      % orthonormal basis orthogonal to the ones vector
P = kron(eye(2), W);
r = max(abs(eig(P'*network(Lx, Lv, T)*P)));
end

function Tlo = largest_stable_T(Lx, Lv, hi)
% Bisection on the root size: an independent check of the closed-form bound.
Ts = linspace(1e-4, hi, 3000);
k = 1;
while disagreement_radius(Lx, Lv, Ts(k)) < 1
    k = k + 1;
end
Tlo = 1e-4;  Tup = Ts(k);
for it = 1:60
    mid = (Tlo + Tup)/2;
    if disagreement_radius(Lx, Lv, mid) < 1
        Tlo = mid;
    else
        Tup = mid;
    end
end
end

function [p11, p22] = mode_variances(alpha, beta, T)
% Steady-state variances (p11, p22) of one mode under unit-intensity noise.
D = (beta - T*alpha)*(4 - 2*T*beta + T*T*alpha);
p11 = (2 - T*beta + T*T*alpha)/(alpha*D);
p22 = 2/D;
end

function V = Vsum(lam, T, kind)
% Sum over modes 2..n of the per-mode variance ('x' positions, 'v' velocities).
V = 0;
for j = 2:numel(lam)
    [p11, p22] = mode_variances(lam(j), lam(j), T);
    if kind == 'x'
        V = V + p11;
    else
        V = V + p22;
    end
end
end

function V = V_lyapunov(L, T, kind)
% The same number from the discrete Lyapunov equation of the whole
% disagreement network, P = F P F' + G G', solved through vec(P).
% With the Control System Toolbox: Pd = dlyap(Fd, Gd*Gd')
n = size(L, 1);
[U, D] = eig(L);
[~, idx] = sort(diag(D));
U = U(:, idx);
P2 = kron(eye(2), U(:, 2:end));
Fd = P2'*network(L, L, T)*P2;
Gd = P2'*[zeros(n); sqrt(T)*eye(n)];
m = size(Fd, 1);
Qd = Gd*Gd';
Pd = reshape((eye(m^2) - kron(Fd, Fd))\Qd(:), m, m);
P = P2*Pd*P2';
if kind == 'x'
    V = trace(P(1:n, 1:n));
else
    V = trace(P(n+1:2*n, n+1:2*n));
end
end

function [xx, yy] = mid_steps(x, y)
% Staircase through the points (x, y) with the steps halfway between the x values.
x = x(:)';  y = y(:)';
mids = (x(1:end-1) + x(2:end))/2;
left = [x(1), mids];
right = [mids, x(end)];
xx = reshape([left; right], 1, []);
yy = reshape([y; y], 1, []);
end
