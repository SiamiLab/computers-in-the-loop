%% Computers in the Loop, Chapter 1: Introduction to Digital Control
% The laser example: an integral controller K/s around the plant 1/(s+a),
% first analog, then run on a computer with sampling period T.
% Milad Siami
%
% The analog design (a = 2, K = 1) gives Y/R = 1/(s+1)^2 for every K.
% On the computer the controller is u_k = u_{k-1} + K T e_{k-1} (rectangle
% rule) and the plant through a zero-order hold is y_{k+1} = p y_k + b u_k,
% with p = e^{-aT} and b = (1 - p)/a (exact at the samples).
% The loop degrades as T grows and is unstable for T > 4/a.
%
% Run this file from the matlab/ folder. Figures go to matlab/figures/.

clear; close all;
here = fileparts(mfilename('fullpath'));
figdir = fullfile(here, 'figures');
if ~exist(figdir, 'dir'), mkdir(figdir); end

a = 2;      % plant 1/(s+a)
K = 1;      % integral gain (K = a^2/4 gives critical damping)

%% Analog design: y(t) = 1 - (1 + t) e^{-t}
yAnalog = @(t) 1 - (1 + t).*exp(-t);
t = linspace(0, 20, 200001);
y = yAnalog(t);
outside = find(abs(y - 1) > 0.02);          % samples outside the 2% band
fprintf('Analog design, a = 2, K = 1:  Y/R = 1/(s+1)^2\n');
fprintf('  2%% settling time                  = %.2f s\n', t(outside(end)));
% expected: 5.83 s
fprintf('  disturbance peak t e^-t at t = 1  = %.3f\n', exp(-1));
% expected: 0.368

%% Digital loop, same K, growing T
% The two closed-loop roots are complex for these T; their product
% is |lambda|^2 = p + b K T.
fprintf('\nDigital loop, same K, growing T\n');
fprintf('   T (s)   step peak   disturbance peak   |lambda|   decay per second\n');
for T = [0.1 0.5 1.0 1.5 2.0 2.5]
    p = exp(-a*T);
    b = (1 - p)/a;
    lam2 = p + b*K*T;
    lam = sqrt(lam2);
    decay = -log(lam)/T;
    if abs(decay) < 5e-4, decay = 0; end     % print 0.000, not -0.000
    [~, ys] = digital_loop(a, K, T, 1, 0, []);   % unit step reference
    [~, yd] = digital_loop(a, K, T, 0, 1, []);   % unit step disturbance
    if abs(lam2 - 1) < 1e-9
        fprintf('   %4.1f    oscillates                         %6.3f      %6.3f\n', T, lam, decay);
    elseif lam2 < 1
        fprintf('   %4.1f    %7.3f     %10.3f         %6.3f      %6.3f\n', ...
                T, max(ys), max(yd), lam, decay);
    else
        fprintf('   %4.1f         grows                         %6.3f      %6.3f\n', T, lam, decay);
    end
end
% expected:
%    0.1      1.000          0.381          0.910       0.945
%    0.5      1.052          0.432          0.725       0.643
%    1.0      1.310          0.491          0.753       0.283
%    1.5      1.703          0.499          0.873       0.090
%    2.0    oscillates                      1.000       0.000
%    2.5         grows                      1.117      -0.044

% The hand table for T = 1 s
[~, ys] = digital_loop(a, K, 1.0, 1, 0, 8);
fprintf('\nT = 1 s, first samples of y_k:');
fprintf(' %.4f', ys(1:7));
fprintf('\n');
% expected: 0 0 0.4323 0.9232 1.2350 1.3104 1.2190
fprintf('Stability limit of the digital loop: T < 4/a = %g s\n', 4/a);
% expected: 2 s

%% Phase margin: delay estimate against the sampled loop
% Analog loop L(s) = 1/(s(s+2)); the computer adds a delay of about T.
wc = sqrt(sqrt(5) - 2);                     % |L(j wc)| = 1
pm0 = 90 - atand(wc/2);
fprintf('\nAnalog loop: crossover %.3f rad/s, phase margin %.1f deg\n', wc, pm0);
% expected: crossover 0.486 rad/s, phase margin 76.3 deg
fprintf('   T (s)   PM, delay estimate   PM of the sampled loop\n');
for T = [0.1 0.5 1.0 1.5 2.0]
    p = exp(-a*T);
    b = (1 - p)/a;
    w = linspace(1e-4, pi/T, 400001);
    z = exp(1j*w*T);
    L = K*T*b ./ ((z - 1).*(z - p));        % D(z) = KT/(z-1), G(z) = b/(z-p)
    [~, ic] = min(abs(abs(L) - 1));         % closest point to |L| = 1
    pm = mod(angle(L(ic))*180/pi + 360, 360) - 180;   % 180 + angle, wrapped
    if abs(pm) < 0.05, pm = 0; end
    fprintf('   %4.1f        %6.1f             %6.1f\n', T, pm0 - wc*T*180/pi, pm);
end
% expected:
%    0.1          73.6               73.5
%    0.5          62.4               61.2
%    1.0          48.5               43.3
%    1.5          34.6               22.9
%    2.0          20.7                0.0

%% Quantization step of a 12-bit converter on +-10 V
q = 20/2^12;
fprintf('\n12-bit converter on +-10 V: q = %.2f mV, rounding error <= %.2f mV\n', ...
        1000*q, 500*q);
% expected: q = 4.88 mV, rounding error <= 2.44 mV

%% Figure: step responses for several T
fig = figure('Color', 'w', 'Position', [100 100 1000 360]);
subplot(1, 2, 1); hold on; box on;
tt = linspace(0, 12, 600);
h = plot(tt, yAnalog(tt), 'k--', 'LineWidth', 1);
labels = {'analog design'};
for T = [0.5 1.0 1.5]
    n = floor(12/T) + 1;
    [k, ys] = digital_loop(a, K, T, 1, 0, n);
    h(end+1) = plot(k*T, ys, 'o-', 'MarkerSize', 3);
    labels{end+1} = sprintf('samples, T = %g s', T);
end
xlabel('t (s)'); ylabel('y'); title('Step response');
legend(h, labels, 'FontSize', 8, 'Location', 'southeast');

subplot(1, 2, 2); hold on; box on;
h = gobjects(0); labels = {};
for T = [2.0 2.5]
    n = floor(40/T) + 1;
    [k, ys] = digital_loop(a, K, T, 1, 0, n);
    h(end+1) = plot(k*T, ys, 'o-', 'MarkerSize', 3);
    labels{end+1} = sprintf('T = %g s', T);
end
xlabel('t (s)'); title('At and past the limit T = 4/a');
legend(h, labels, 'FontSize', 8, 'Location', 'northwest');

print(fig, fullfile(figdir, 'ch01_laser_step.png'), '-dpng', '-r150');
fprintf('\nSaved figures/ch01_laser_step.png\n');

%% Local function
function [k, y] = digital_loop(a, K, T, r, d, n)
% Simulate the sampled loop from rest. r: step reference, d: step
% disturbance at the plant input (both start at k = 0).
% Returns the sample numbers k = 0, 1, ... and y_k.
p = exp(-a*T);
b = (1 - p)/a;
if isempty(n), n = ceil(40/T) + 2; end
y = zeros(1, n);
u = zeros(1, n);
e = zeros(1, n);
for i = 1:n-1                       % i = k + 1 (MATLAB indices start at 1)
    e(i) = r - y(i);
    if i >= 2
        u(i) = u(i-1) + K*T*e(i-1); % rectangle rule, one sample old
    end
    y(i+1) = p*y(i) + b*(u(i) + d); % ZOH + plant, exact at the samples
end
k = 0:n-1;
end
