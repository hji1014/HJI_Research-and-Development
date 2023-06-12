%% Initialize
clc; close all; clear;

%% SET PARAMETERS for STFT
fs = 400;
R = 80;                % R: window length (200ms)
window = hamming(R);   % hamming window, length R
N = 2^16;              % N: FFT resolution
L = ceil(R*0.1);       % L: number of non-overlap samples
overlap = R - L;       % Overlap = 50% of window length

[s, f, t] = spectrogram(x, window, overlap, N, fs, 'yaxis');

s = final_all_ersp_avg;

figure
%imagesc(t,f,log10(abs(s)));
imagesc(t(77:107),f(1:4917),log10(abs(s(1:4917, :))));      % t : 0-600ms
colormap(jet)
axis xy
xlabel('time')
ylabel('frequency')
title('SPECTROGRAM, R = 80')
colorbar