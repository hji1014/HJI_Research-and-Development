%% Initialize
clc; close all; clear;

%% Data import
a = load('200_invalid.mat');
b = load('200_valid.mat');

data1 = a.Data;
data2 = b.Data;

eeg_1000 = mean(data2(:, :, :), 3);

eeg_1000_sample = eeg_1000(1, :);

%figure(1)
%plot(eeg_1000_sample)
%figure(2)
%plot(eeg_1000_sample-mean(eeg_1000_sample(1:200)))
%plot(t, eeg_1000_sample)

%% LPF 20Hz

Fs = 400;
N = 400;

xn = eeg_1000_sample;
tn = linspace(-600, 1400, Fs*2);

Nf = 5;         % 3차 필터
Fp = 20;        % 20Hz LPF
Ap = 1;         % 통과대역 리플 = 1dB
As = 60;        % 저지대역 감쇠량 = 60dB

d = designfilt('lowpassiir','FilterOrder',Nf,'PassbandFrequency',Fp,...
    'PassbandRipple',Ap,'StopbandAttenuation',As,'SampleRate',Fs);

xfilter = filter(d,xn);

plot(tn, xn, tn, xfilter)

% 일반 필터 지연 확인
grpdelay(d,N,Fs)
fvtool(d)

xfiltfilt = filtfilt(d, xn);
xfiltfilt_BC = xfiltfilt - mean(xfiltfilt(1:160));

figure(1)
plot(tn, xn, tn, xfilter)
hold on
plot(tn, xfiltfilt, 'r', 'linewidth', 2)
hold off


%% Baseline correction
%baseline : -600 ms ~ -200 ms -> tn(1:160)

baseline_correction_eeg_1000_sample = eeg_1000_sample- mean(eeg_1000_sample(1:160));













