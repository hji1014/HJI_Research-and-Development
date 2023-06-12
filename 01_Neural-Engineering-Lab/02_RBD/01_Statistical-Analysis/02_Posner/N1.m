%% 주의사항
% 1. path 설정
% 2. sprintf 경로 바꾸기
% 3. baseline 설정(cue_onset/target_onset)

%% Initialize
clc; close all; clear;

%% LPF 20Hz -> Baseline Correction(before stimulus) -> Individual Avg(Ensemble Avg)

%cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_Posner_paradigm\control'
path = 'C:\Users\Nelab_001\Documents\MATLAB\RBD_Posner_paradigm\control';   % 'control' path
%path = 'C:\Users\Nelab_001\Documents\MATLAB\RBD_Posner_paradigm\RBD';   % 'RBD' path
%cd(path)
List = dir(path);
% List_name = {List.name};
% List_name = List_name(3:end);
subject_num = length(List);


for folder_num = 3:subject_num
    %% Path setting
    
    % sprintf 경로 바꾸기
    % control
    foldername = List(folder_num).name;
    present_valid_200 = sprintf('./control/%s/200_valid.mat', foldername);
    present_invalid_200 = sprintf('./control/%s/200_invalid.mat', foldername);
    present_valid_1000 = sprintf('./control/%s/1000_valid.mat', foldername);
    present_invalid_1000 = sprintf('./control/%s/1000_invalid.mat', foldername);
    % RBD
    %present_valid_200 = sprintf('./RBD/%s/200_valid.mat', foldername);
    %present_invalid_200 = sprintf('./RBD/%s/200_invalid.mat', foldername);
    %present_valid_1000 = sprintf('./RBD/%s/1000_valid.mat', foldername);
    %present_invalid_1000 = sprintf('./RBD/%s/1000_invalid.mat', foldername);
    
    % SOA 200ms
    valid_200 = importdata(present_valid_200);
    invalid_200 = importdata(present_invalid_200);
    
    % SOA 1000ms
    valid_1000 = importdata(present_valid_1000);
    invalid_1000 = importdata(present_invalid_1000);
    
    %% LPF 20Hz hyperparameter setting
    
    Fs = 400;
    N = 400;

    tn_200 = linspace(-600, 1400, Fs*2); %tn_200 = -0.600:1/400:1.400-1/400;
    tn_1000 = linspace(-1600, 1400, Fs*3);

    Nf = 5;         % 7차 필터
    Fp = 20;        % 20Hz LPF
    Ap = 1;         % 통과대역 리플 = 1dB
    As = 60;        % 저지대역 감쇠량 = 60dB

    d = designfilt('lowpassiir','FilterOrder',Nf,'PassbandFrequency',Fp, ...
        'PassbandRipple',Ap,'StopbandAttenuation',As,'SampleRate',Fs);

    % 일반 필터 지연 확인
    % grpdelay(d,N,Fs)

    %xfiltfilt = filtfilt(d, xn);
    %xfiltfilt_BC = xfiltfilt - mean(xfiltfilt(1:160));

    % prestimulus criteria
    % cue -> -600~-200ms / target -> -600~0ms / paper -> -100~0ms
    cue_onset_200 = 160;
    target_onest_200 = 240;

    cue_onset_1000 = 240;
    target_onset_1000 = 640;
    paper_onset = 600:640;

    
    %% Filtering and Baseline Correction
    
    % valid_200
    
    for i = 1:60        % each ch
        valid_200_ch = valid_200(i, :, :);
        for j = 1:size(valid_200_ch, 3)       % each trial
            valid_200_trial = squeeze(valid_200_ch(:, :, j));
      
            % filtering
            xn = valid_200_trial;
            xfiltfilt = filtfilt(d, xn);
      
            % baseline correction
            xfilt_BC = xfiltfilt - mean(xfiltfilt(paper_onset));
      
            % integration
            valid_200(i, :, j) = xfilt_BC;
      
        end
    end

    % invalid_200
    
    for m = 1:60        % each ch
        invalid_200_ch = invalid_200(m, :, :);
        for n = 1:size(invalid_200_ch, 3)       % each trial
            invalid_200_trial = squeeze(invalid_200_ch(:, :, n));
      
            % filtering
            xn = invalid_200_trial;
            xfiltfilt = filtfilt(d, xn);
      
            % baseline correction
            xfilt_BC = xfiltfilt - mean(xfiltfilt(paper_onset));
      
            % integration
            invalid_200(m, :, n) = xfilt_BC;
      
        end
    end
    
    % valid_1000

    for i = 1:60        % each ch
        valid_1000_ch = valid_1000(i, :, :);
        for j = 1:size(valid_1000_ch, 3)       % each trial
            valid_1000_trial = squeeze(valid_1000_ch(:, :, j));
      
            % filtering
            xn = valid_1000_trial;
            xfiltfilt = filtfilt(d, xn);
      
            % baseline correction
            xfilt_BC = xfiltfilt - mean(xfiltfilt(paper_onset));
      
            % integration
            valid_1000(i, :, j) = xfilt_BC;
      
        end
    end

    % invalid_1000

    for i = 1:60        % each ch
        invalid_1000_ch = invalid_1000(i, :, :);
        for j = 1:size(invalid_1000_ch, 3)       % each trial
            invalid_1000_trial = squeeze(invalid_1000_ch(:, :, j));
      
            % filtering
            xn = invalid_1000_trial;
            xfiltfilt = filtfilt(d, xn);
      
            % baseline correction
            xfilt_BC = xfiltfilt - mean(xfiltfilt(paper_onset));
      
            % integration
            invalid_1000(i, :, j) = xfilt_BC;
      
        end
    end
    
    %% Ensemble Average(individual_subject)
    
    valid_200_chavg = zeros(60, 800);
    invalid_200_chavg = zeros(60, 800);
    valid_1000_chavg = zeros(60, 1200);
    invalid_1000_chavg = zeros(60, 1200);
    
    % N1 avg
    valid_200_chavg_N1 = zeros(60, 1);
    invalid_200_chavg_N1 = zeros(60, 1);
    valid_1000_chavg_N1 = zeros(60, 1);
    invalid_1000_chavg_N1 = zeros(60, 1);

    % valid_200
    for a = 1:60
        valid_200_avg = mean(squeeze(valid_200(a, :, :)), 2);
        valid_200_chavg(a, :) = valid_200_avg;
        % N1 mean amplitude
        valid_200_chavg_N1(a, :) = mean(valid_200_avg(321:341), 1);
    end

    % invalid_200
    for b = 1:60
        invalid_200_avg = mean(squeeze(invalid_200(b, :, :)), 2);
        invalid_200_chavg(b, :) = invalid_200_avg;
        % N1 mean amplitude
        invalid_200_chavg_N1(b, :) = mean(invalid_200_avg(321:341), 1);
    end

    % valid_1000
    for a = 1:60
        valid_1000_avg = mean(squeeze(valid_1000(a, :, :)), 2);
        valid_1000_chavg(a, :) = valid_1000_avg;
        % N1 mean amplitude
        valid_1000_chavg_N1(a, :) = mean(valid_1000_avg(721:740), 1);
    end

    % invalid_1000
    for b = 1:60
        invalid_1000_avg = mean(squeeze(invalid_1000(b, :, :)), 2);
        invalid_1000_chavg(b, :) = invalid_1000_avg;
        % N1 mean amplitude
        invalid_1000_chavg_N1(b, :) = mean(invalid_1000_avg(721:741), 1);
    end
    
    %% Flotting
    %figure(1)
    %for plotid = 1 : 15
    %    subplot(3, 5, plotid);
    %    plot(tn_1000, valid_1000_chavg(plotid, :), tn_1000, invalid_1000_chavg(plotid, :))
    %    legend('valid 1000', 'invalid 1000')
    %    xlim([-1600 1400])
    %    xticks([-1600 0 1400])
    %    xline(0, '--k');
    %end
    
    %% Save file '.mat'
    save(foldername, 'valid_200_chavg', 'invalid_200_chavg', 'valid_1000_chavg','invalid_1000_chavg'...
        , 'valid_200_chavg_N1', 'invalid_200_chavg_N1', 'valid_1000_chavg_N1', 'invalid_1000_chavg_N1')
    
end