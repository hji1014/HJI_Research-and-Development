%% ERSP size(=s_size) 바꿔줘야함 - 총 6번
%% Initialize
clc; close all; clear;
%% 플롯용

% SOA 1000ms
%valid_1000 = load('1000_valid.mat').Data;

%a = squeeze(valid_1000(:, :, 1));
%x = a(1, :);

%fs = 400;
%R = 80;                % R: window length (200ms)
%window = hamming(R);   % hamming window, length R
%N = 2^16;              % N: FFT resolution
%L = ceil(R*0.1);       % L: number of non-overlap samples
%overlap = R - L;       % Overlap = 50% of window length

%[s, f, t] = spectrogram(x, window, overlap, N, fs, 'yaxis');

%figure
%imagesc(t(77:107),f(1:4917),log10(abs(s(1:4917, :))));      % t : 0-600ms,f : 30Hz
%colormap(jet)
%axis xy
%xlabel('time')
%ylabel('frequency')
%title('SPECTROGRAM, R = 80')
%colorbar

%% SET PARAMETERS for STFT
fs = 400;
%R = 80;                % R: window length (200ms)
R = 40;
window = hamming(R);   % hamming window, length R
N = 2^10;              % N: FFT resolution -> 원하는 수로 다시하기 f resolution 결정
L = ceil(R*0.5);       % L: number of non-overlap samples
overlap = R - L;       % Overlap = 50% of window length


%cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_Posner_paradigm\control'
%path = 'C:\Users\Nelab_001\Documents\MATLAB\RBD_Posner_paradigm\control';   % 'control' path
path = 'C:\Users\Nelab_001\Documents\MATLAB\RBD_Posner_paradigm\RBD';   % 'RBD' path
%cd(path)
List = dir(path);
% List_name = {List.name};
% List_name = List_name(3:end);
subject_num = length(List);     % subject_num - 3 + 1 : 실제 피험자수
real_num = subject_num-3+1;

%% Valid ERSP
%% ERSP 그리기 (실험마다 ERSP size 바꿔줘야함 - 3번)

final_all_ersp = zeros(real_num, 513, 59);       % ERSP size 바꿔주기 (32769 x 141에서 다른걸로)


for folder_num = 3:subject_num
%% Path setting
    
    % sprintf 경로 바꾸기
    % control
    foldername = List(folder_num).name;
    %present_valid_200 = sprintf('./control/%s/200_valid.mat', foldername);
    %present_invalid_200 = sprintf('./control/%s/200_invalid.mat', foldername);
    %present_valid_1000 = sprintf('./control/%s/1000_valid.mat', foldername);
    %present_invalid_1000 = sprintf('./control/%s/1000_invalid.mat', foldername);
    % RBD
    present_valid_200 = sprintf('./RBD/%s/200_valid.mat', foldername);
    present_invalid_200 = sprintf('./RBD/%s/200_invalid.mat', foldername);
    present_valid_1000 = sprintf('./RBD/%s/1000_valid.mat', foldername);
    present_invalid_1000 = sprintf('./RBD/%s/1000_invalid.mat', foldername);
    
    % SOA 1000ms
    valid_1000 = importdata(present_valid_1000);
    invalid_1000 = importdata(present_invalid_1000);
    
    valid_s_all_ch_avg_all = zeros(60, 513, 59);                 % ERSP size 바꿔주기 (32769 x 141에서 다른걸로)
    
    for i = 1:60        % each ch
        valid_1000_ch = squeeze(valid_1000(i, :, :));
        valid_s_all_ch = zeros(size(valid_1000, 3), 513, 59);    % ERSP size 바꿔주기 (32769 x 141에서 다른걸로)    
        for j = 1:size(valid_1000, 3)       % each trial
            valid_1000_trial = squeeze(valid_1000_ch(:, j));     % 1 x 1200
            [valid_s, f, t] = spectrogram(valid_1000_trial, window, overlap, N, fs, 'yaxis');       % valid_s : 32769 x 141
            valid_s_all_ch(j, :, :) = valid_s;                   % for (j)문 끝나면 -> trial x 32769 x 141
        end
        %위에까지 ㅇㅋ
        valid_s_all_ch_avg = squeeze(mean(valid_s_all_ch, 1));   % 32769 x 141 (n번 피험자 m번 ch 모든 trial avg)
        valid_s_all_ch_avg_all(i, :, :) = valid_s_all_ch_avg;    % 60 x 32769 x 141
    end
    
    ind_valid_s = squeeze(mean(valid_s_all_ch_avg_all, 1));      % 32769 x 141 -> 개인 모든 trial, ch 평균
    final_all_ersp(folder_num-2, :, :) = ind_valid_s;            % real_num x 32769 x 141
    
end

final_all_ersp_avg = squeeze(mean(final_all_ersp, 1));          % 전체 피험자 ERSP avg
control_valid = final_all_ersp_avg;                             % valid averaged ERSP






%% Invalid ERSP
%% ERSP 그리기 (실험마다 ERSP size 바꿔줘야함 - 3번)

final_all_ersp = zeros(real_num, 513, 59);       % ERSP size 바꿔주기 (32769 x 141에서 다른걸로)


for folder_num = 3:subject_num
%% Path setting
    
    % sprintf 경로 바꾸기
    % control
    foldername = List(folder_num).name;
    %present_valid_200 = sprintf('./control/%s/200_valid.mat', foldername);
    %present_invalid_200 = sprintf('./control/%s/200_invalid.mat', foldername);
    %present_valid_1000 = sprintf('./control/%s/1000_valid.mat', foldername);
    %present_invalid_1000 = sprintf('./control/%s/1000_invalid.mat', foldername);
    % RBD
    present_valid_200 = sprintf('./RBD/%s/200_valid.mat', foldername);
    present_invalid_200 = sprintf('./RBD/%s/200_invalid.mat', foldername);
    present_valid_1000 = sprintf('./RBD/%s/1000_valid.mat', foldername);
    present_invalid_1000 = sprintf('./RBD/%s/1000_invalid.mat', foldername);
    
    % SOA 1000ms
    %valid_1000 = importdata(present_valid_1000);
    %invalid_1000 = importdata(present_invalid_1000);
    valid_1000 = importdata(present_invalid_1000);
    
    valid_s_all_ch_avg_all = zeros(60, 513, 59);                 % ERSP size 바꿔주기 (32769 x 141에서 다른걸로)
    
    for i = 1:60        % each ch
        valid_1000_ch = squeeze(valid_1000(i, :, :));
        valid_s_all_ch = zeros(size(valid_1000, 3), 513, 59);    % ERSP size 바꿔주기 (32769 x 141에서 다른걸로)    
        for j = 1:size(valid_1000, 3)       % each trial
            valid_1000_trial = squeeze(valid_1000_ch(:, j));     % 1 x 1200
            [valid_s, f, t] = spectrogram(valid_1000_trial, window, overlap, N, fs, 'yaxis');       % valid_s : 32769 x 141
            valid_s_all_ch(j, :, :) = valid_s;                   % for (j)문 끝나면 -> trial x 32769 x 141
        end
        %위에까지 ㅇㅋ
        valid_s_all_ch_avg = squeeze(mean(valid_s_all_ch, 1));   % 32769 x 141 (n번 피험자 m번 ch 모든 trial avg)
        valid_s_all_ch_avg_all(i, :, :) = valid_s_all_ch_avg;    % 60 x 32769 x 141
    end
    
    ind_valid_s = squeeze(mean(valid_s_all_ch_avg_all, 1));      % 32769 x 141 -> 개인 모든 trial, ch 평균
    final_all_ersp(folder_num-2, :, :) = ind_valid_s;            % real_num x 32769 x 141
    
end

final_all_ersp_avg = squeeze(mean(final_all_ersp, 1));          % 전체 피험자 ERSP avg
control_invalid = final_all_ersp_avg;                           % valid averaged ERSP

%% plotting

s1 = control_invalid;
s2 = control_valid;

figure(1)
imagesc(t(32:44),f(1:81),log10(abs(s1(1:81,:))));             % t:0-600ms, f:30Hz limit
%imagesc(t(77:107),f(1:4917),log10(abs(s(1:4917, :))));      % t : 0-600ms
colormap(jet)
axis xy
xlabel('time')
ylabel('frequency')
title('SPECTROGRAM, R = 80')
colorbar
%caxis([-1 1]);

figure(2)
imagesc(t(32:44),f(1:81),log10(abs(s2(1:81,:))));             % t:0-600ms, f:30Hz limit
%imagesc(t(77:107),f(1:4917),log10(abs(s(1:4917, :))));      % t : 0-600ms
colormap(jet)
axis xy
xlabel('time')
ylabel('frequency')
title('SPECTROGRAM, R = 80')
colorbar
%caxis([-1 1]);

figure(3)
imagesc(t(32:44),f(1:81),log10(abs(s1(1:81,:)))-log10(abs(s2(1:81,:))));             % t:0-600ms, f:30Hz limit
%imagesc(t(77:107),f(1:4917),log10(abs(s(1:4917, :))));      % t : 0-600ms
colormap(jet)
axis xy
xlabel('time')
ylabel('frequency')
title('SPECTROGRAM, R = 80')
colorbar
%caxis([-1 1]);