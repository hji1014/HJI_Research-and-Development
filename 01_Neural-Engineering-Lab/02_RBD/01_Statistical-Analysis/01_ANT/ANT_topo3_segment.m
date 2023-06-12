% ANT_topo3_segment
% -> rbd 62명이라 쪼개서 3d topo로 변환해야함

%% Initialize
clc; close all; clear;
tic

%% convert to 3d topo -> data명 변경, 저장할 변수명 변경
load('C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\chanlocs_60.mat')
Fs = 200;
tn = linspace(-400, 2200, Fs*2.6);      % -400ms~2200ms
baseline = 41:80;                       % -200ms~0ms

% import data
data = importdata('rbd_input.mat');
EEG = data.input;
EEG = permute(EEG, [3 1 2]);
gt = data.ground_truth;

% filtering, baseline correction

Nf = 5;         % 7차 필터
Fp = 20;        % 20Hz LPF
Ap = 1;         % 통과대역 리플 = 1dB
As = 60;        % 저지대역 감쇠량 = 60dB

d = designfilt('lowpassiir','FilterOrder',Nf,'PassbandFrequency',Fp, ...
    'PassbandRipple',Ap,'StopbandAttenuation',As,'SampleRate',Fs);

for i = 1:size(EEG, 1)
    for j = 1:size(EEG, 2)
        trial = squeeze(EEG(i, j, :));
        trial_f = filtfilt(d, trial);
        trial_bc = trial_f - mean(trial_f(baseline));
        trial_bc = normalize(trial_bc);                 % z-score normalization
        EEG(i, j, :) = trial_bc;
    end
end

%EEG = EEG(1:10000, :, :);
EEG = EEG(10001:17401, :, :);
%EEG = EEG(10001:20000, :, :);
%EEG = EEG(20001:30000, :, :);
%EEG = EEG(30001:34802, :, :);

% topo
topo_3d = zeros(size(EEG, 1), 67, 67, 14);

for k = 1:size(EEG, 1)                  % trial
    for l = 1:14                        % time(3D)
        eeg = squeeze(EEG(k, :, :));
        time_range = (81+20*(l-1)):(100+20*(l-1));
        eeg_avg = mean(eeg(:, time_range), 2);
        [grid_or_val, plotrad_or_grid] = topoplot(eeg_avg, chanlocs,'noplot','on');
        topo = plotrad_or_grid;
        topo_3d(k, :, :, l) = topo;
    end
end

%ANT_rbd_topo_3d = topo_3d;
topo_3d_2 = topo_3d;
%ground_truth = gt;

%save('ANT_rbd_topo_3d.mat', 'ANT_rbd_topo_3d', 'ground_truth', '-v7.3')
save('topo_3d_2.mat', 'topo_3d_2', '-v7.3')

toc

%% 쪼갠거 합치기
a = importdata('topo_3d_1.mat');
b = importdata('topo_3d_2.mat');
c = importdata('topo_3d_3.mat');
d = importdata('topo_3d_4.mat');

%ANT_con_topo_3d = cat(1, a, b);
%ANT_rbd_topo_3d = cat(1, a, b, c, d);
ANT_rbd_topo_3d = cat(1, a, b);

%save('ANT_con_topo_3d.mat', 'ANT_con_topo_3d', 'ground_truth', '-v7.3')
save('ANT_rbd_topo_3d.mat', 'ANT_rbd_topo_3d', 'ground_truth', '-v7.3')
