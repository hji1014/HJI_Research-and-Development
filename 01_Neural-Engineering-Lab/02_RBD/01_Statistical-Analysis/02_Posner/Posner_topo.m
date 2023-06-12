%% 주의사항
% 1. path 설정
% 2. sprintf 경로 바꾸기
% 3. save cd 바꾸기

%% Initialize
clc; close all; clear;

%% 

path = 'C:\Users\Nelab_001\Documents\MATLAB\RBD_Posner_paradigm\control';   % 'control' path
%path = 'C:\Users\Nelab_001\Documents\MATLAB\RBD_Posner_paradigm\RBD';   % 'RBD' path
%cd(path)
List = dir(path);
% List_name = {List.name};
% List_name = List_name(3:end);
subject_num = length(List);

%%
all_ch_val_200 = [];
all_ch_inval_200 = [];
all_ch_val_1000 = [];
all_ch_inval_1000 = [];

all_ch_tr_val_200 = [];
all_ch_tr_inval_200 = [];
all_ch_tr_val_1000 = [];
all_ch_tr_inval_1000 = [];

all_sub_RBD = [];
all_sub_CON = [];

% parameters
Fs = 400; % after down sampling
tn_200 = linspace(-600, 1400, Fs*2); %tn_200 = -0.600:1/400:1.400-1/400;
tn_1000 = linspace(-1600, 1400, Fs*3);
base_line_200 = 200:240;   % -100ms~0ms
base_line_1000 = 600:640;
N1_200 = 320:340;
N1_1000 = 720:740;

%%

Nf = 5;         % 7차 필터
Fp = 20;        % 20Hz LPF
Ap = 1;         % 통과대역 리플 = 1dB
As = 60;        % 저지대역 감쇠량 = 60dB

d = designfilt('lowpassiir','FilterOrder',Nf,'PassbandFrequency',Fp, ...
    'PassbandRipple',Ap,'StopbandAttenuation',As,'SampleRate',Fs);

%%
load('chan_loc_snuh_60ch.mat') % 분석하고자 하는 전극 채널의 theta, radius 값 로드
theta = {chanlocs.theta};
radius = {chanlocs.radius};
for i = 1:size(theta,2)
    t(i) = theta{i};
    r(i) = radius{i};
end
t2 = t/180*pi;
r2 = r/(max(r)+0.01);
x = r2 .* cos(t2);
y = r2 .* sin(t2);
x2 = 0.5*sqrt(2 + 2*x*sqrt(2) + x.^2 - y.^2) - 0.5*sqrt(2 - 2*x*sqrt(2) + x.^2 - y.^2);
y2 = 0.5*sqrt(2 + 2*y*sqrt(2) - x.^2 + y.^2) - 0.5*sqrt(2 - 2*y*sqrt(2) - x.^2 + y.^2);
[xq,yq] = meshgrid(linspace(-1,1,67), linspace(-1,1,67));


%%
for folder_num = 3:subject_num
    %% Path setting
    
    cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_Posner_paradigm'
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
    
    % valid_1000
    for i = 1:size(valid_1000, 3)
        for j = 1:60
            trial = valid_1000(j, :, i);
            
            % filtering
            trial_f = filtfilt(d, trial);
            
            % baseline correction
            trial_bc = trial_f - mean(trial(base_line_1000));
            
            % N1
            trial_N1 = mean(trial_bc(N1_1000));
            
            % all_ch
            all_ch_val_1000(j, 1) = trial_N1;
            
            
        end
        % square interpolation
        si_N1 = single(griddata(x2,y2,all_ch_val_1000,xq,yq,'v4'));
        si_N1_90 = rot90(si_N1);
                        
        % all_trial
        all_ch_tr_val_1000(i, :, :) = si_N1_90;
    end
    
    % invalid_1000
    for i = 1:size(invalid_1000, 3)
        for j = 1:60
            trial = invalid_1000(j, :, i);
            
            % filtering
            trial_f = filtfilt(d, trial);
            
            % baseline correction
            trial_bc = trial_f - mean(trial(base_line_1000));
            
            % N1
            trial_N1 = mean(trial_bc(N1_1000));
            
            % all_ch
            all_ch_inval_1000(j, 1) = trial_N1;
            
        end
        % square interpolation
        si_N1 = single(griddata(x2,y2,all_ch_inval_1000,xq,yq,'v4'));
        si_N1_90 = rot90(si_N1);
                        
        % all_trial
        all_ch_tr_inval_1000(i, :, :) = si_N1_90;
    end
    
    % valid_200
    for i = 1:size(valid_200, 3)
        for j = 1:60
            trial = valid_200(j, :, i);
            
            % filtering
            trial_f = filtfilt(d, trial);
            
            % baseline correction
            trial_bc = trial_f - mean(trial(base_line_200));
            
            % N1
            trial_N1 = mean(trial_bc(N1_200));
            
            % all_ch
            all_ch_val_200(j, 1) = trial_N1;
            
            
        end
        % square interpolation
        si_N1 = single(griddata(x2,y2,all_ch_val_200,xq,yq,'v4'));
        si_N1_90 = rot90(si_N1);
                        
        % all_trial
        all_ch_tr_val_200(i, :, :) = si_N1_90;
    end
    
    % invalid_200
    for i = 1:size(invalid_200, 3)
        for j = 1:60
            trial = invalid_200(j, :, i);
            
            % filtering
            trial_f = filtfilt(d, trial);
            
            % baseline correction
            trial_bc = trial_f - mean(trial(base_line_200));
            
            % N1
            trial_N1 = mean(trial_bc(N1_200));
            
            % all_ch
            all_ch_inval_200(j, 1) = trial_N1;
            
            
        end
        % square interpolation
        si_N1 = single(griddata(x2,y2,all_ch_inval_200,xq,yq,'v4'));
        si_N1_90 = rot90(si_N1);
                        
        % all_trial
        all_ch_tr_inval_200(i, :, :) = si_N1_90;
    end

    %% Save file '.mat'
    %m = '.mat';
    %name = strcat(foldername, m);
    %cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_Posner_paradigm\ERSP\CON_ERSP'
    %cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_Posner_paradigm\ERSP\RBD_ERSP'
    %cd 'E:\ERSP\CON_ERSP'
    %cd 'E:\ERSP\RBD_ERSP'
    
    RBD = cat(1, all_ch_tr_val_1000, all_ch_tr_inval_1000, all_ch_tr_val_200, all_ch_tr_inval_200);
    
    
    %all_sub_RBD = cat(1, all_sub_RBD, RBD);
    all_sub_CON = cat(1, all_sub_CON, RBD);
    
    
end

%save('RBD_topo', 'all_sub_RBD', '-v7.3')
save('CON_topo', 'all_sub_CON', '-v7.3')
