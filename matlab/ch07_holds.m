%% Computers in the Loop, Chapter 7: Data Reconstruction and Data Holds
% Gain and phase of the zero-order, fractional-order and first-order holds,
% the equal-lag frequency, the FOH peak, the images of a held sinusoid, and
% the three holds reconstructing one sampled signal.
% Milad Siami
%
%   G_ho(s) = (1 - e^{-Ts})/s
%   G_h1(s) = (1 + Ts)/T [(1 - e^{-Ts})/s]^2
%   G_hb(s) = (1 - beta e^{-Ts})(1 - e^{-Ts})/s + beta/(T s^2) (1 - e^{-Ts})^2
%
% Run this file from the matlab/ folder. Figures go to matlab/figures/.

clear; close all;
here = fileparts(mfilename('fullpath'));
figdir = fullfile(here, 'figures');
if ~exist(figdir, 'dir'), mkdir(figdir); end

T = 1;                  % results are quoted in units of T
ws = 2*pi/T;

Gzoh  = @(w) (1 - exp(-T*1j*w))./(1j*w);
Gfrac = @(w, beta) (1 - beta*exp(-T*1j*w)).*(1 - exp(-T*1j*w))./(1j*w) ...
                   + beta./(T*(1j*w).^2).*(1 - exp(-T*1j*w)).^2;
Gfoh  = @(w) (1 + T*1j*w)/T .* ((1 - exp(-T*1j*w))./(1j*w)).^2;

%% Gain and phase at wT = 0.5 and at ws/2
fprintf('Gain (in units of T) and phase (deg)\n');
fprintf('                     wT = 0.5              w = ws/2\n');
names = {'ZOH          ', 'beta = 0.5   ', 'FOH (beta=1) '};
holds = {Gzoh, @(w) Gfrac(w, 0.5), Gfoh};
for i = 1:3
    G1 = holds{i}(0.5/T);
    G2 = holds{i}(pi/T);
    p2 = angle(G2)*180/pi;
    if p2 > 0, p2 = p2 - 360; end    % report the lag as a negative angle
    fprintf('  %s   %.3f T, %6.1f       %.3f T, %7.1f\n', names{i}, ...
            abs(G1)/T, angle(G1)*180/pi, abs(G2)/T, p2);
end
% expected:
%   ZOH             0.990 T,  -14.3       0.637 T,   -90.0
%   beta = 0.5      1.036 T,   -7.9       0.976 T,  -102.0
%   FOH (beta=1)    1.095 T,   -2.1       1.336 T,  -107.7
wTest = [0.3 1.7];
fprintf('  the fractional hold with beta = 1 equals the FOH: %d\n', ...
        max(abs(Gfrac(wTest, 1) - Gfoh(wTest))) < 1e-12);
% expected: 1 (true)

%% Equal phase lags and the peak gain of the FOH
% The ZOH lags by wT/2, the FOH by wT - atan(wT).
x = fzero(@(v) v/2 - (v - atan(v)), [0.5 3]);
fprintf('\nEqual phase lag where wT/2 = wT - atan(wT): wT = %.2f, w = %.2f ws\n', x, x/(2*pi));
% expected: wT = 2.33, w = 0.37 ws
w = linspace(0.01, ws, 200001);
g = abs(Gfoh(w))/T;
[gmax, imax] = max(g);
fprintf('Peak gain of the FOH: %.2f T at w = %.2f ws\n', gmax, w(imax)/ws);
% expected: 1.59 T at w = 0.34 ws

%% Images of a held sinusoid at w0 = ws/10
% Components at |w0 + n ws| with amplitude |sin(v)/v|, v = pi (w0 + n ws)/ws
w0 = ws/10;
fprintf('\nZOH on sin(w0 t), w0 = ws/10: amplitude of the component at |w0 + n ws|\n');
for nn = [0 -1 1]
    v = pi*(w0 + nn*ws)/ws;
    fprintf('  n = %2d, |w0 + n ws| = %.1f ws: %.3f\n', nn, abs(w0 + nn*ws)/ws, abs(sin(v)/v));
end
% expected: n = 0: 0.984;  n = -1: 0.109;  n = 1: 0.089

%% Low-frequency delay of the fractional hold: (1 - beta) T/2
for beta = [0 0.5 1]
    wl = 1e-3;
    fprintf('  beta = %g: low-frequency delay = %.4f T (formula (1 - beta)/2 = %.4f)\n', ...
            beta, -angle(Gfrac(wl, beta))/wl, (1 - beta)/2);
end
% expected: 0.5000, 0.2500, 0.0000 (formula and computed value agree)

%% Figure: reconstruction from samples, and the hold magnitudes
Ts = 0.5;
f = @(t) 1.1 + 0.8*sin(0.9*t) + 0.12*t;
tk = Ts*(0:20);
ek = f(tk);
t = linspace(0, 10, 2001);
k = min(floor(t/Ts), numel(tk) - 1);   % sample number of the current period
zoh = ek(k + 1);                       % +1: MATLAB indices start at 1
prev = ek(max(k - 1, 0) + 1);
slope = (ek(k + 1) - prev)/Ts;
foh = ek(k + 1) + slope.*(t - tk(k + 1));
half = ek(k + 1) + 0.5*slope.*(t - tk(k + 1));

fig = figure('Color', 'w', 'Position', [100 100 1000 340]);
subplot(1, 2, 1); hold on; box on;
h1 = plot(t, f(t), 'k', 'LineWidth', 1);
h2 = plot(t, zoh);
h3 = plot(t, half);
h4 = plot(t, foh);
plot(tk, ek, 'ko', 'MarkerSize', 3);
xlabel('t (s)'); title('Reconstruction from samples, T = 0.5 s');
legend([h1 h2 h3 h4], {'e(t)', 'ZOH', '\beta = 0.5', 'FOH'}, 'FontSize', 8, 'Location', 'northwest');
subplot(1, 2, 2); hold on; box on;
wn = linspace(0.001, 2*ws, 2000);
h1 = plot(wn/ws, abs(Gzoh(wn))/T);
h2 = plot(wn/ws, abs(Gfoh(wn))/T);
xlabel('\omega / \omega_s'); ylabel('|G_h| / T'); title('Hold magnitudes');
legend([h1 h2], {'ZOH', 'FOH'}, 'FontSize', 8);
print(fig, fullfile(figdir, 'ch07_holds.png'), '-dpng', '-r150');
fprintf('\nSaved figures/ch07_holds.png\n');
