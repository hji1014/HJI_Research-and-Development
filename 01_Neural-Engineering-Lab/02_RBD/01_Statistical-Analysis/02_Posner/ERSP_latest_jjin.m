%% 주의사항
% 1. path 설정
% 2. sprintf 경로 바꾸기
% 3. save cd 바꾸기

%% Initialize
clc; close all; clear;

%% 

%path = 'C:\Users\Nelab_001\Documents\MATLAB\RBD_Posner_paradigm\control';   % 'control' path
path = 'C:\Users\Nelab_001\Documents\MATLAB\RBD_Posner_paradigm\RBD';   % 'RBD' path
%cd(path)
List = dir(path);
% List_name = {List.name};
% List_name = List_name(3:end);
subject_num = length(List);

%%

ERSP_valid_1000 = [];
ERSP_invalid_1000 = [];

% parameters
Fs = 200; % after down sampling
tn_200 = linspace(-600, 1400, Fs*2); %tn_200 = -0.600:1/400:1.400-1/400;
tn_1000 = linspace(-1600, 1400, Fs*3);
base_line = 300:320;   % -100ms~0ms

% cwt configuration
Ts = 1/Fs;
F_upper_bound = 100;
num = 5;

%%
for folder_num = 3:subject_num
    %% Path setting
    
    cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_Posner_paradigm'
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
    
    % Down sampling(200Hz)
    % valid_1000
    for i = 1:60
        for j = 1:size(valid_1000, 3)
            trial = valid_1000(i, :, j);
            trial_down = downsample(trial, 2);
            
            % ERSP(cwt)
            y=cwt_cmor_norm_var_cycd(trial_down, Ts, F_upper_bound, num);
            % = cwt_cmor_norm_var_cyc(trial_down, 1/Fs);
            
            % Baseline correction
            %y_bc = y - mean(y(:, base_line), 2);
            
            ERSP_valid_1000(i, :, :, j) = y;            % size = (60, f, samples, trial)
        end
    end
    
    % invalid_1000
    for i = 1:60
        for j = 1:size(invalid_1000, 3)
            trial = invalid_1000(i, :, j);
            trial_down = downsample(trial, 2);
            
            % ERSP(cwt)
            y=cwt_cmor_norm_var_cycd(trial_down, Ts, F_upper_bound, num);
            % = cwt_cmor_norm_var_cyc(trial_down, 1/Fs);
            
            % Baseline correction
            %y_bc = y - mean(y(:, base_line), 2);
            
            ERSP_invalid_1000(i, :, :, j) = y;
        end
    end

    %% Save file '.mat'
    m = '.mat';
    name = strcat(foldername, m);
    %cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_Posner_paradigm\ERSP\CON_ERSP'
    %cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_Posner_paradigm\ERSP\RBD_ERSP'
    %cd 'E:\ERSP\CON_ERSP'
    cd 'E:\ERSP\RBD_ERSP'
    save(name, 'ERSP_valid_1000', 'ERSP_invalid_1000', '-v7.3')
    
    ERSP_valid_1000 = [];
    ERSP_invalid_1000 = [];
end

%%

imagesc(abs(y2 - mean(y2(:, base_line), 2)));
figure(1)
imagesc(abs(a(3:30, :)));
axis xy;
colormap('jet');
colorbar
figure(2)
imagesc(abs(b(3:30, :)));
axis xy;
colormap('jet');
colorbar