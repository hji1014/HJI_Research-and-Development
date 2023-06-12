% ANT_topo3

%% Initialize
clc; close all; clear;

%% 조건 별 trial 결합(input, ground_truth) -> path, save name 설정

path = 'C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\con\1.Incongruent';

List = dir(path);
subject_num = length(List);
%load('chanlocs_60.mat')

input = [];
ground_truth = [];

for folder_num = 3:subject_num
    subject_name = List(folder_num).name;
    file_address = sprintf('./%s', subject_name);
    data = importdata(file_address);
    EEG = data.EEG_data;
        
    label = folder_num - 2;
    label_vector = zeros(size(EEG, 3), 1);
    label_vector(:) = label;
    
    input = cat(3, input, EEG);
    ground_truth = cat(1, ground_truth, label_vector);

end

save('rbd_05.mat', 'input', 'ground_truth', '-v7.3')

%% 모든 조건 결합, down sampling(400to200) -> path, save name 설정

path = 'D:\ANT_3D_CNN\ANT_cat\rbd';

List = dir(path);
subject_num = length(List);
%load('chanlocs_60.mat')

input = [];
ground_truth = [];

% for folder_num = 3:subject_num
for folder_num = 3:5
    subject_name = List(folder_num).name;
    file_address = sprintf('./%s', subject_name);
    data = importdata(file_address);
    EEG = data.input;
    gt = data.ground_truth;
    
    input = cat(3, input, EEG);
    ground_truth = cat(1, ground_truth, gt);
    
end

input_downsampled = [];

for i = 1:size(input, 1)
    for j = 1:size(input, 3)
        down = downsample(input(i, :, j), 2);
        input_downsampled(i, :, j) = down;
    end
end

input = input_downsampled;

save('C:/Users/Nelab_001/Desktop/rbd_input.mat', 'input', 'ground_truth', '-v7.3')

%% convert to 3d topo -> data명 변경, 저장할 변수명 변경
tic

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
        EEG(i, j, :) = trial_bc;
    end
end

% topo
topo_3d = zeros(size(EEG, 1), 67, 67, 14);
%topo_3d = zeros(10000, 67, 67, 14);

for k = 1:size(EEG, 1)                  % trial
%for k = 1:10000                  % trial
    for l = 1:14                        % time(3D)
        eeg = squeeze(EEG(k, :, :));
        time_range = (81+20*(l-1)):(100+20*(l-1));
        eeg_avg = mean(eeg(:, time_range), 2);
        [grid_or_val, plotrad_or_grid] = topoplot(eeg_avg, chanlocs,'noplot','on');
        topo = plotrad_or_grid;
        topo_3d(k, :, :, l) = topo;
    end
end

toc

ANT_rbd_topo_3d = topo_3d;
ground_truth = gt;

save('ANT_rbd_topo_3d.mat', 'ANT_rbd_topo_3d', 'ground_truth', '-v7.3')
