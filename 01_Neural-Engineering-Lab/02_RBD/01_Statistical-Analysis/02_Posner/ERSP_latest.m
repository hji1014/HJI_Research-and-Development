%% 주의사항
% 1. path 설정
% 2. sprintf 경로 바꾸기
% 3. baseline 설정(cue_onset/target_onset)

%% Initialize
clc; close all; clear;

%% 

%cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_Posner_paradigm\control'
%path = 'C:\Users\Nelab_001\Documents\MATLAB\RBD_Posner_paradigm\control';   % 'control' path
path = 'C:\Users\Nelab_001\Documents\MATLAB\RBD_Posner_paradigm\RBD';   % 'RBD' path
%cd(path)
List = dir(path);
% List_name = {List.name};
% List_name = List_name(3:end);
subject_num = length(List);

all_sub_ensemble_avg_valid_200 = zeros(subject_num-2, 60, 800);
all_sub_ensemble_avg_invalid_200 = zeros(subject_num-2, 60, 800);
all_sub_ensemble_avg_valid_1000 = zeros(subject_num-2, 60, 1200);
all_sub_ensemble_avg_invalid_1000 = zeros(subject_num-2, 60, 1200);

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
    
    % SOA 200ms
    valid_200 = importdata(present_valid_200);
    invalid_200 = importdata(present_invalid_200);
    
    % SOA 1000ms
    valid_1000 = importdata(present_valid_1000);
    invalid_1000 = importdata(present_invalid_1000);
    
    %% Ensemble Average(individual_subject)
    
    valid_200_trial_avg = zeros(60, 800);
    invalid_200_trial_avg = zeros(60, 800);
    valid_1000_trial_avg = zeros(60, 1200);
    invalid_1000_trial_avg = zeros(60, 1200);

    % valid_200
    for a = 1:60
        valid_200_avg = mean(squeeze(valid_200(a, :, :)), 2).';
        valid_200_trial_avg(a, :) = valid_200_avg;
        for i = 1:subject_num-2
            all_sub_ensemble_avg_valid_200(i, a, :) = valid_200_avg;
        end
    end
    
    % invalid_200
    for b = 1:60
        invalid_200_avg = mean(squeeze(invalid_200(b, :, :)), 2).';
        invalid_200_trial_avg(b, :) = invalid_200_avg;
        for i = 1:subject_num-2
            all_sub_ensemble_avg_invalid_200(i, b, :) = invalid_200_avg;
        end
    end
    
    % valid_1000
    for c = 1:60
        valid_1000_avg = mean(squeeze(valid_1000(c, :, :)), 2).';
        valid_1000_trial_avg(c, :) = valid_1000_avg;
        for i = 1:subject_num-2
            all_sub_ensemble_avg_valid_1000(i, c, :) = valid_1000_avg;
        end
    end
    
    % valid_200
    for d = 1:60
        invalid_1000_avg = mean(squeeze(invalid_1000(d, :, :)), 2).';
        invalid_1000_trial_avg(d, :) = invalid_1000_avg;
        for i = 1:subject_num-2
            all_sub_ensemble_avg_invalid_1000(i, d, :) = invalid_1000_avg;
        end
    end
    
    %% Save file '.mat'
    save('ensemble_avg.mat', 'all_sub_ensemble_avg_valid_200', 'all_sub_ensemble_avg_invalid_200'...
        , 'all_sub_ensemble_avg_valid_1000','all_sub_ensemble_avg_invalid_1000')
    
end

%% importing AVGs
% Initialize
clc; close all; clear;

% path
path = 'C:\Users\Nelab_001\Documents\MATLAB\RBD_Posner_paradigm\ERSP\CON_ERSP';   % 'control' path
%path = 'C:\Users\Nelab_001\Documents\MATLAB\RBD_Posner_paradigm\RBD';   % 'RBD' path
cd(path)

% parameters
Fs = 400;
tn_200 = linspace(-600, 1400, Fs*2); %tn_200 = -0.600:1/400:1.400-1/400;
tn_1000 = linspace(-1600, 1400, Fs*3);
cue_onset_200 = 160;
target_onest_200 = 240;
cue_onset_1000 = 240;
target_onset_1000 = 640;
paper_onset = 600:640;

% Grand Average

ensemble_avg = importdata('ensemble_avg.mat');

all_sub_ensemble_avg_valid_200 = ensemble_avg.all_sub_ensemble_avg_valid_200;
all_sub_ensemble_avg_invalid_200 = ensemble_avg.all_sub_ensemble_avg_invalid_200;
all_sub_ensemble_avg_valid_1000 = ensemble_avg.all_sub_ensemble_avg_valid_1000;
all_sub_ensemble_avg_invalid_1000 = ensemble_avg.all_sub_ensemble_avg_invalid_1000;

grand_avg_valid_200 = squeeze(mean(all_sub_ensemble_avg_valid_200, 1));
grand_avg_invalid_200 = squeeze(mean(all_sub_ensemble_avg_invalid_200, 1));
grand_avg_valid_1000 = squeeze(mean(all_sub_ensemble_avg_valid_1000, 1));
grand_avg_invalid_1000 = squeeze(mean(all_sub_ensemble_avg_invalid_1000, 1));

%% CWT
% y=cwt_cmor_norm_var_cyc(data,Ts)와 같은 형태로 입력함.
% Ts는 1/fs와 같음
% num : 주파수 윈도우 사이즈 변경 주기

ERSP_t = 641:880; % 0ms~600ms
grand_avg = mean(grand_avg_valid_1000(:, ERSP_t), 1).';

Ts = 1/Fs;
F_upper_bound = 50;
num = 5;
y=cwt_cmor_norm_var_cycd(grand_avg, Ts, F_upper_bound, num);

% baseline correction


imagesc(abs(y));
axis xy;
colormap('jet');
colorbar