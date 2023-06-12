%% initialization
clear; close all; clc;

%% data loading
read_Intan_RHS2000_file
% fs : 30000 Hz

% % HPF 250 Hz
% for j=1:32
%     amplifier_data_HPF(j, :) = HPF(amplifier_data(j, :), 30000, 300);
% end

% 64 채널 2번째
amplifier_data = amplifier_data(33:64, :);

% BPF [300 Hz, 5000 Hz]
for i=1:32
    amplifier_data_BPF(i, :) = bandpass(amplifier_data(i, :), [300 5000], 30000);
end

% for i = 1:32
%     figure(1)
%     subplot(4, 8, i)
%     plot(t, amplifier_data(i, :))
% end
% for i = 1:32
%     figure(2)
%     subplot(4, 8, i)
%     plot(t, amplifier_data_BPF(i, :))
% end

% downsampling 30000 to 5000
amplifier_data_BPF_down = downsample(amplifier_data_BPF.', 6).';
t_down = downsample(t.', 6).';
t_down_zero = t_down - t_down(1);

xlimit = [0 60];
ylimit = [-150 150];

for i = 1:32
    if i<=8
        figure(1)
        subplot(8, 1, i)
        plot(t_down_zero, amplifier_data_BPF_down(i, :), 'k')
        xlim(xlimit)
        ylim(ylimit)
    elseif i>8 && i<17
        figure(2)
        subplot(8, 1, i-8)
        plot(t_down_zero, amplifier_data_BPF_down(i, :), 'k')
        xlim(xlimit)
        ylim(ylimit)
    elseif i>16 && i<25
        figure(3)
        subplot(8, 1, i-16)
        plot(t_down_zero, amplifier_data_BPF_down(i, :), 'k')
        xlim(xlimit)
        ylim(ylimit)
    elseif i>24
        figure(4)
        subplot(8, 1, i-24)
        plot(t_down_zero, amplifier_data_BPF_down(i, :), 'k')
        xlim(xlimit)
        ylim(ylimit)
    end
end

% 변수 여러개 만들기
for n = 1:32
    eval(sprintf('ch%d = amplifier_data_BPF_down(%d, :);',n, n))
end

Fs = 5000;
time_window = 1;
threshold = -40;
time_range = 1:500;     % 0.1 s

% local minimum index 찾기
local_min_idx = [];
for i = 1:size(amplifier_data_BPF_down, 1)
    TF = islocalmin(amplifier_data_BPF_down(i, :));
    for j = 1:size(amplifier_data_BPF_down, 2)
        if amplifier_data_BPF_down(i, j) > threshold
            TF(j) = 0;
        end
    end
    local_min_idx(i, :) = TF;
end
local_min_idx = logical(local_min_idx);
local_min_num = length(find(local_min_idx));
local_min_idx_num = find(local_min_idx);

%% 일단 3번 채널, threshold = -40 or -60
a = amplifier_data_BPF_down(5, :);
ylimit = [-150 150];
threshold = 60;

%% *************** spike sorting based on negative threshold ***************
tf_n = islocalmin(a);
for i=1:length(a)
    if a(i) > -threshold || a(i) < -200
        tf_n(i) = 0;
    end
end

local_min_idx = logical(tf_n);
local_min_num = length(find(tf_n));
local_min_idx_num = find(tf_n);

% figure(2)       % local min 확인
% plot(t_down_zero, a, t_down_zero(tf), a(tf), 'r*')

all_spike = [];
for i=1:local_min_num
    spike_range = (local_min_idx_num(i)-5):(local_min_idx_num(i)+14);
    spike = a(1, spike_range);
    all_spike(:, i) = spike;
end
all_spike_mean = mean(all_spike, 2);

figure(5)        % 채널 별 전체 spike 및 평균 spike 확인
plot(all_spike)
ylim(ylimit)
hold on
plot(all_spike_mean,'k', 'LineWidth', 3)
hold off

%% *************** spike sorting based on positive threshold ***************
[pks, locs] = findpeaks(a);
tf_p = zeros(1, length(a));
tf_p(1, locs) = 1;

for i=1:length(a)
    if a(i) < threshold || a(i) > 200
        tf_p(i) = 0;
    end
end

local_max_idx = logical(tf_p);
local_max_num = length(find(tf_p));
local_max_idx_num = find(tf_p);

all_spike = [];
for i=1:local_max_num
    spike_range = (local_max_idx_num(i)-5):(local_max_idx_num(i)+14);
    spike = a(1, spike_range);
    all_spike(:, i) = spike;
end
all_spike_mean = mean(all_spike, 2);

figure(6)        % 채널 별 전체 spike 및 평균 spike 확인
plot(all_spike)
ylim(ylimit)
hold on
plot(all_spike_mean,'k', 'LineWidth', 3)
hold off

%% *************** spike sorting based on squared threshold ***************
tf_s = tf_n + tf_p;

for i=1:length(a)
    if (a(i))^2 < (threshold)^2 || a(i) > 200 || a(i) < -200
        tf_s(i) = 0;
    end
end

max_idx = logical(tf_s);
max_num = length(find(tf_s));
max_idx_num = find(tf_s);

all_spike = [];
for i=1:max_num
    spike_range = (max_idx_num(i)-5):(max_idx_num(i)+14);
    spike = a(1, spike_range);
    all_spike(:, i) = spike;
end
all_spike_mean = mean(all_spike, 2);

figure(7)        % 채널 별 전체 spike 및 평균 spike 확인
plot(all_spike)
ylim([-200 250])
