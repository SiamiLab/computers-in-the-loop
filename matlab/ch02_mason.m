%% Computers in the Loop, Chapter 2: Block Diagrams and Mason's Gain Formula
% The robot-arm position loop, and Mason's gain formula checked against the
% node equations on Example 2, Example 3 and the rate-feedback loop.
% Milad Siami
%
% Mason's formula is coded for any signal flow graph whose branch gains are
% numbers (or transfer functions evaluated at one value of s). The forward
% paths and the loops are listed by depth-first search; then Delta and the
% cofactors Delta_k are built from the loops that do not touch.
%
% Run this file from the matlab/ folder. Figures go to matlab/figures/.

clear; close all;
here = fileparts(mfilename('fullpath'));
figdir = fullfile(here, 'figures');
if ~exist(figdir, 'dir'), mkdir(figdir); end

%% Robot arm, reflected to the motor shaft
Jm = 0.01;  Bm = 0.01;          % motor inertia and friction
Jarm = 4;   Barm = 4;           % arm inertia and friction
n = 0.1;                        % gear ratio
R = 1;  Kt = 0.5;  Km = 0.5;    % armature resistance, torque and back-emf constants
J = Jm + n^2*Jarm;
B = Bm + n^2*Barm;
K = n*Kt/(R*J);
a = (R*B + Kt*Km)/(R*J);
fprintf('Robot arm, reflected to the motor shaft\n');
fprintf('  J_eq = %.3f, B_eq = %.3f, K = %.3f, a = %.3f  ->  G(s) = %g/(s(s+%g))\n', J, B, K, a, K, a);
% expected: J_eq = 0.050, B_eq = 0.050, K = 1.000, a = 6.000  ->  G(s) = 1/(s(s+6))

Gc = 25;
den = [1, a, K*Gc];             % closed loop 25/(s^2 + 6s + 25)
poles = roots(den);
wn = sqrt(den(3));
fprintf('  G_c = 25: closed loop 25/(s^2 + %gs + 25), poles %s, wn = %g rad/s, zeta = %g\n', ...
        a, num2str(poles.', 4), wn, a/(2*wn));
% expected: poles -3 +- 4j, wn = 5 rad/s, zeta = 0.6

L = 0.01;                                        % keep the inductance
motor = conv([L, R], [J, B]) + [0, 0, Kt*Km];    % (R + Ls)(Js + B) + Kt Km
fprintf('  with L = 0.01 H: (R+Ls)(Js+B) + KtKm = %g (s^2 + %gs + %g), poles of G: 0, %s\n', ...
        motor(1), motor(2)/motor(1), motor(3)/motor(1), num2str(round(100*roots(motor).')/100));
% expected: 0.0005 (s^2 + 101s + 600), poles of G: 0, -94.66, -6.34
fprintf('  equal real closed-loop poles: G_c = a^2/4 = %g\n', a^2/4);
% expected: 9

%% Example 2: two forward paths, two loops that do not touch
% Branch list: {from, to, gain}. Here G1..G5 = 1, G6 = 2, H1 = 1, H2 = 3.
G1 = 1; G2 = 1; G3 = 1; G4 = 1; G5 = 1; G6 = 2; H1 = 1; H2 = 3;
br = {'R','x1',G1; 'x1','x2',G2; 'x2','x1',-H1; 'x2','x3',G3; ...
      'R','x3',G6; 'x3','x4',G4; 'x4','x3',-H2; 'x4','C',G5};
[Tm, paths, loops, names] = mason(br, 'R', 'C');
fprintf('\nExample 2 with G1..G5 = 1, G6 = 2, H1 = 1, H2 = 3\n');
fprintf('  forward paths:');
for i = 1:numel(paths)
    fprintf('  %s (gain %g)', strjoin(names(paths(i).nodes), '-'), paths(i).gain);
end
fprintf('\n');
% expected: R-x1-x2-x3-x4-C (gain 1)  R-x3-x4-C (gain 2)
fprintf('  loops:        ');
for i = 1:numel(loops)
    fprintf('  %s (gain %g)', strjoin(names(loops(i).nodes), '-'), loops(i).gain);
end
fprintf('\n');
% expected: x1-x2 (gain -1)  x3-x4 (gain -3)
fprintf('  Mason: T_CR = %.4f (5/8 = %g),  node equations: %.4f\n', ...
        real(Tm), 5/8, real(node_equations(br, 'R', 'C')));
% expected: Mason 0.6250, node equations 0.6250

%% Example 3: three interlaced loops, against the block-diagram reduction
for s = [0.7 + 1.3j, 2.0]
    G1 = 2/(s + 1);  G2 = 3/(s + 2);  G3 = 1/s;  G4 = 4/(s + 3);
    H1 = 0.5;  H2 = s/(s + 5);  H3 = 1;
    br3 = {'R','E1',1; 'E1','E2',G1; 'E2','E3',G2; 'E3','y',G3; 'y','C',G4; ...
           'C','E3',H1; 'y','E2',-H2; 'C','E1',-H3};
    [Tm, ~, loops3] = mason(br3, 'R', 'C');
    Tred = G1*G2*G3*G4/(1 - G3*G4*H1 + G2*G3*H2 + G1*G2*G3*G4*H3);
    fprintf('\nExample 3 at s = %s: %d loops; Mason %s, reduction %s, node equations %s\n', ...
            cstr(s, 1), numel(loops3), cstr(Tm, 6), cstr(Tred, 6), ...
            cstr(node_equations(br3, 'R', 'C'), 6));
end
% expected: s = 0.7+1.3j: 3 loops; all three give -0.718224-0.865874j
%           s = 2:        3 loops; all three give  0.180645

%% Rate (tachometer) feedback inside a unity position loop, plant 1/(s(s+1))
% R -> e -> [K] -> x -> [1/(s+1)] -> v -> [1/s] -> C, v -> x with -1, C -> e with -1
Kg = 4;
for s = [0.5 + 0.5j, 3.0]
    brt = {'R','e',1; 'e','x',Kg; 'x','v',1/(s + 1); 'v','C',1/s; 'v','x',-1; 'C','e',-1};
    Tm = mason(brt, 'R', 'C');
    fprintf('Rate feedback, K = %g, s = %s: Mason %s,  K/(s^2+2s+K) = %s\n', ...
            Kg, cstr(s, 1), cstr(Tm, 6), cstr(Kg/(s*s + 2*s + Kg), 6));
end
% expected: s = 0.5+0.5j: 0.733945-0.220183j (both);  s = 3: 0.210526 (both)

%% Figure: arm step responses, with and without the inductance
% With the Control System Toolbox: step(tf(25, den), t)
t = linspace(0, 2.5, 1000);
y2 = step_tf(25, den, t);
numG3 = n*Kt;  denG3 = conv(motor, [1 0]);       % G(s) with L kept (third order)
y3 = step_tf(Gc*numG3, denG3 + [0, 0, 0, Gc*numG3], t);
fig = figure('Color', 'w', 'Position', [100 100 600 340]);
hold on; box on;
h1 = plot(t, y2);
h2 = plot(t, y3, '--');
xlabel('t (s)'); ylabel('arm angle / command');
title('Robot-arm position loop, G_c = 25');
legend([h1 h2], {'L neglected: 25/(s^2+6s+25)', 'L = 0.01 H (third order)'}, ...
       'FontSize', 8, 'Location', 'southeast');
print(fig, fullfile(figdir, 'ch02_arm_step.png'), '-dpng', '-r150');
fprintf('\nPeak overshoot: %.1f%% (L neglected), %.1f%% (L kept)\n', ...
        100*(max(y2) - 1), 100*(max(y3) - 1));
% expected: 9.5% (L neglected), 10.1% (L kept)
fprintf('Saved figures/ch02_arm_step.png\n');

%% Local functions
function [Tm, paths, loops, names] = mason(br, src, snk)
% Mason's gain formula T = sum_k P_k Delta_k / Delta for the graph br.
[from, to, g, names] = graph_lists(br);
isrc = find(strcmp(names, src));
isnk = find(strcmp(names, snk));
% forward paths by depth-first search
paths = struct('nodes', {}, 'gain', {});
paths = find_paths(from, to, g, isrc, isnk, isrc, 1, paths);
% loops: each one is found once, starting from its lowest-numbered node
loops = struct('nodes', {}, 'gain', {});
for s0 = 1:numel(names)
    loops = find_loops(from, to, g, s0, s0, s0, 1, loops);
end
D = delta(loops);
num = 0;
for k = 1:numel(paths)
    % cofactor: Delta of the loops that do not touch path k
    keep = false(1, numel(loops));
    for i = 1:numel(loops)
        keep(i) = isempty(intersect(loops(i).nodes, paths(k).nodes));
    end
    num = num + paths(k).gain*delta(loops(keep));
end
Tm = num/D;
end

function [from, to, g, names] = graph_lists(br)
% Node names sorted alphabetically; branches as row vectors of node numbers.
names = sort(unique([br(:,1); br(:,2)]))';
nb = size(br, 1);
from = zeros(1, nb); to = zeros(1, nb); g = zeros(1, nb);
for e = 1:nb
    from(e) = find(strcmp(names, br{e,1}));
    to(e) = find(strcmp(names, br{e,2}));
    g(e) = br{e,3};
end
end

function paths = find_paths(from, to, g, node, snk, visited, gain, paths)
if node == snk
    paths(end+1) = struct('nodes', visited, 'gain', gain);
    return
end
for e = find(from == node)
    v = to(e);
    if ~any(visited == v)
        paths = find_paths(from, to, g, v, snk, [visited v], gain*g(e), paths);
    end
end
end

function loops = find_loops(from, to, g, start, node, visited, gain, loops)
for e = find(from == node)
    v = to(e);
    if v == start
        loops(end+1) = struct('nodes', visited, 'gain', gain*g(e));
    elseif ~any(visited == v) && v > start
        loops = find_loops(from, to, g, start, v, [visited v], gain*g(e), loops);
    end
end
end

function D = delta(loops)
% Delta = 1 - sum L_i + sum of products of two non-touching loops - ...
D = 1;
nL = numel(loops);
for r = 1:nL
    if nL == 1
        combos = 1;
    else
        combos = nchoosek(1:nL, r);
    end
    for c = 1:size(combos, 1)
        idx = combos(c, :);
        ok = true;
        for i = 1:r
            for j = i+1:r
                if ~isempty(intersect(loops(idx(i)).nodes, loops(idx(j)).nodes))
                    ok = false;
                end
            end
        end
        if ok
            D = D + (-1)^r*prod([loops(idx).gain]);
        end
    end
end
end

function x = node_equations(br, src, snk)
% Solve x = A x directly (each node is the sum of what enters it), x_src = 1.
[from, to, g, names] = graph_lists(br);
nn = numel(names);
A = zeros(nn);
for e = 1:numel(g)
    A(to(e), from(e)) = A(to(e), from(e)) + g(e);
end
isrc = find(strcmp(names, src));
isnk = find(strcmp(names, snk));
M = eye(nn) - A;
M(isrc, :) = 0;
M(isrc, isrc) = 1;
rhs = zeros(nn, 1);
rhs(isrc) = 1;
sol = M\rhs;
x = sol(isnk);
end

function y = step_tf(num, den, t)
% Unit-step response of num(s)/den(s) on the uniform grid t (t(1) = 0),
% exact at the grid points: the state equations are sampled with expm.
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
% A complex number as text, e.g. -0.718224-0.865874j
s = sprintf(['%.' num2str(d) 'f%+.' num2str(d) 'fj'], real(z), imag(z));
end
