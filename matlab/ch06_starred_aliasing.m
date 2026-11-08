%% Computers in the Loop, Chapter 6: The Starred Transform, Aliasing, and the Zero-Order Hold
% Starred transforms by series and by residues (Examples 1, 2 and 3),
% aliasing and hidden oscillations, and the frequency response of the ZOH.
% Milad Siami
%
% E*(s) = sum_k e(kT) e^{-kTs}.
%
% Run this file from the matlab/ folder. Figures go to matlab/figures/.

clear; close all;
here = fileparts(mfilename('fullpath'));
figdir = fullfile(here, 'figures');
if ~exist(figdir, 'dir'), mkdir(figdir); end

K = 0:2999;                          % sample numbers used in the series
starred = @(samples, s, T) sum(samples .* exp(-K(1:numel(samples))*T*s));

%% Examples 1 and 2: the unit step and 1 - e^{-t}, T = 0.5 s
T = 0.5;
fprintf('Starred transforms, T = 0.5 s, at two values of s\n');
for s = [0.8 + 2.0j, 2.0]
    stepSeries = starred(ones(1, 3000), s, T);
    fprintf('  s = %s: unit step  series %s   1/(1 - e^-Ts) = %s\n', ...
            cstr(s, 1), cstr(stepSeries, 6), cstr(1/(1 - exp(-T*s)), 6));
    f = starred(1 - exp(-K*T), s, T);
    res = 1/(1 - exp(-T*s)) - 1/(1 - exp(-T*(s + 1)));    % residues at s = 0 and s = -1
    fprintf('             1 - e^-t   series %s   residue formula %s\n', cstr(f, 6), cstr(res, 6));
end
% expected:
%   s = 0.8+2j: unit step 0.879785-0.778030j;  1 - e^-t: -0.195112-0.306768j
%   s = 2:      unit step 1.581977;            1 - e^-t:  0.294760
%   (series and closed form agree in every case)

%% Example 3: e(t) = [1 - e^{-(t-1)}] u(t-1), a delay of 2T with T = 0.5 s
kT = K*T;
eTrue = zeros(size(kT));
eTrue(kT >= 1) = 1 - exp(-(kT(kT >= 1) - 1));
eWrong = 1 - exp(-(kT - 1));         % what the residues of e^{-s}/(s(s+1)) describe
fprintf('\nExample 3, e(t) = [1 - e^{-(t-1)}] u(t-1), T = 0.5 s\n');
fprintf('  true samples          :'); fprintf(' %.4f', eTrue(1:6)); fprintf('\n');
% expected: 0 0 0 0.3935 0.6321 0.7769
fprintf('  residues on E(s) as is:'); fprintf(' %.4f', eWrong(1:6)); fprintf('   (wrong at k = 0, 1)\n');
% expected: -1.7183 -0.6487 0 0.3935 0.6321 0.7769
s = 1.3 + 0.7j;
closedForm = (1 - exp(-0.5))*exp(-1.5*s)/((1 - exp(-0.5*s))*(1 - exp(-0.5*(s + 1))));
fixed = exp(-2*T*s)*(1/(1 - exp(-T*s)) - 1/(1 - exp(-T*(s + 1))));   % delay taken out first
fprintf('  at s = %s: series %s, closed form %s, delay taken out first %s\n', ...
        cstr(s, 1), cstr(starred(eTrue, s, T), 6), cstr(closedForm, 6), cstr(fixed, 6));
% expected: 0.004324-0.145726j for all three
fprintf('  first nonzero sample e(1.5) = 1 - e^-0.5 = %.4f\n', 1 - exp(-0.5));
% expected: 0.3935

%% Aliasing and hidden oscillations
T = 0.1;
k = 0:10;
s7 = cos(2*pi*7*k*T);
s3 = cos(2*pi*3*k*T);
fprintf('\nAliasing at fs = 10 Hz: max |cos(2pi 7 kT) - cos(2pi 3 kT)| = %.2e\n', max(abs(s7 - s3)));
% expected: about 2e-15 (round-off only)
fprintf('  folding rule: 7 Hz shows up at |7 - 10| = %d Hz\n', abs(7 - 10));
% expected: 3 Hz
fprintf('  cos(2 pi t) at fs = 1 Hz gives the samples'); fprintf(' %g', round(cos(2*pi*(0:4))*1e6)/1e6);
fprintf(' like the constant 1\n');
% expected: 1 1 1 1 1
hiddenSamples = round(sin(pi*(0:5))*1e12)/1e12;
hiddenSamples(hiddenSamples == 0) = 0;             % print 0, not -0
fprintf('  hidden oscillation sin(pi t/T): samples'); fprintf(' %g', hiddenSamples); fprintf('\n');
% expected: 0 0 0 0 0 0

%% Frequency response of the zero-order hold, T = 0.1 s
T = 0.1;
ws = 2*pi/T;
Gho = @(w) (1 - exp(-1j*w*T))./(1j*w);
fprintf('\nZOH, T = %g s (ws = %.1f rad/s)\n', T, ws);
% expected: ws = 62.8 rad/s
for fHz = [1 5]
    G = Gho(2*pi*fHz);
    fprintf('  %3.0f Hz: |G|/T = %.3f, phase = %.1f deg\n', fHz, abs(G)/T, angle(G)*180/pi);
end
% expected: 1 Hz: 0.984, -18.0 deg;  5 Hz: 0.637, -90.0 deg
fprintf('  gain at ws/2 = 2T/pi = %.3f T\n', 2/pi);
% expected: 0.637 T

%% Figure: aliasing and the ZOH magnitude
fig = figure('Color', 'w', 'Position', [100 100 1000 340]);
subplot(1, 2, 1); hold on; box on;
tt = linspace(0, 0.5, 1000);
h1 = plot(tt, cos(2*pi*7*tt));
h2 = plot(tt, cos(2*pi*3*tt), '--');
h3 = plot(k(1:6)*T, s3(1:6), 'ko');
xlabel('t (s)'); title('Aliasing: same samples');
legend([h1 h2 h3], {'7 Hz', '3 Hz', 'samples, fs = 10 Hz'}, 'FontSize', 8, 'Location', 'southwest');
subplot(1, 2, 2); box on;
w = linspace(1e-6, 3*ws, 2000);
plot(w/ws, abs(Gho(w))/T);
xlabel('\omega / \omega_s'); ylabel('|G_{ho}(j\omega)| / T'); title('ZOH magnitude');
print(fig, fullfile(figdir, 'ch06_aliasing_zoh.png'), '-dpng', '-r150');
fprintf('\nSaved figures/ch06_aliasing_zoh.png\n');

%% Local function
function s = cstr(z, d)
% A complex number as text, e.g. 0.879785-0.778030j
s = sprintf(['%.' num2str(d) 'f%+.' num2str(d) 'fj'], real(z), imag(z));
end
