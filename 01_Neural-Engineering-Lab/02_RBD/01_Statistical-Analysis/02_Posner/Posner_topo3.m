%% 주의사항
% 1. path 설정
% 2. sprintf 경로 바꾸기
% 3. save cd 바꾸기

%% Initialize
clc; close all; clear;

tic
%%

path = 'C:\Users\Nelab_001\Documents\MATLAB\RBD_Posner_paradigm\control';   % 'control' path
%path = 'C:\Users\Nelab_001\Documents\MATLAB\RBD_Posner_paradigm\RBD';   % 'RBD' path
List = dir(path);
subject_num = length(List);
load('chan_loc_snuh_60ch.mat')

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

SOA1000_1 = 641:660;    % 0~50ms
SOA1000_2 = 661:680;
SOA1000_3 = 681:700;
SOA1000_4 = 701:720;
SOA1000_5 = 721:740;
SOA1000_6 = 741:760;
SOA1000_7 = 761:780;
SOA1000_8 = 781:800;
SOA1000_9 = 801:820;
SOA1000_10 = 821:840;   %450ms-500ms

SOA200_1 = 241:260;     % 0~50ms
SOA200_2 = 261:280;
SOA200_3 = 281:300;
SOA200_4 = 301:320;
SOA200_5 = 321:340;
SOA200_6 = 341:360;
SOA200_7 = 361:380;
SOA200_8 = 381:400;
SOA200_9 = 401:420;
SOA200_10 = 421:440;   %450ms-500ms

%%

Nf = 5;         % 7차 필터
Fp = 20;        % 20Hz LPF
Ap = 1;         % 통과대역 리플 = 1dB
As = 60;        % 저지대역 감쇠량 = 60dB

d = designfilt('lowpassiir','FilterOrder',Nf,'PassbandFrequency',Fp, ...
    'PassbandRipple',Ap,'StopbandAttenuation',As,'SampleRate',Fs);

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
            trial_1 = mean(trial_bc(SOA1000_1));
            trial_2 = mean(trial_bc(SOA1000_2));
            trial_3 = mean(trial_bc(SOA1000_3));
            trial_4 = mean(trial_bc(SOA1000_4));
            trial_5 = mean(trial_bc(SOA1000_5));
            trial_6 = mean(trial_bc(SOA1000_6));
            trial_7 = mean(trial_bc(SOA1000_7));
            trial_8 = mean(trial_bc(SOA1000_8));
            trial_9 = mean(trial_bc(SOA1000_9));
            trial_10 = mean(trial_bc(SOA1000_10));
            
            % all_ch
            all_ch_val_1000(j, 1) = trial_1;
            all_ch_val_1000(j, 2) = trial_2;
            all_ch_val_1000(j, 3) = trial_3;
            all_ch_val_1000(j, 4) = trial_4;
            all_ch_val_1000(j, 5) = trial_5;
            all_ch_val_1000(j, 6) = trial_6;
            all_ch_val_1000(j, 7) = trial_7;
            all_ch_val_1000(j, 8) = trial_8;
            all_ch_val_1000(j, 9) = trial_9;
            all_ch_val_1000(j, 10) = trial_10;
                       
        end
        
        for k = 1:10
            [grid_or_val, plotrad_or_grid] = topoplot(all_ch_val_1000(:, k), chanlocs,'noplot','on');
            topo = flipud(plotrad_or_grid);
            all_ch_tr_val_1000(i, :, :, k) = topo;
        end
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
            trial_1 = mean(trial_bc(SOA1000_1));
            trial_2 = mean(trial_bc(SOA1000_2));
            trial_3 = mean(trial_bc(SOA1000_3));
            trial_4 = mean(trial_bc(SOA1000_4));
            trial_5 = mean(trial_bc(SOA1000_5));
            trial_6 = mean(trial_bc(SOA1000_6));
            trial_7 = mean(trial_bc(SOA1000_7));
            trial_8 = mean(trial_bc(SOA1000_8));
            trial_9 = mean(trial_bc(SOA1000_9));
            trial_10 = mean(trial_bc(SOA1000_10));
            
            % all_ch
            all_ch_inval_1000(j, 1) = trial_1;
            all_ch_inval_1000(j, 2) = trial_2;
            all_ch_inval_1000(j, 3) = trial_3;
            all_ch_inval_1000(j, 4) = trial_4;
            all_ch_inval_1000(j, 5) = trial_5;
            all_ch_inval_1000(j, 6) = trial_6;
            all_ch_inval_1000(j, 7) = trial_7;
            all_ch_inval_1000(j, 8) = trial_8;
            all_ch_inval_1000(j, 9) = trial_9;
            all_ch_inval_1000(j, 10) = trial_10;
            
        end
        
        for k = 1:10
            [grid_or_val, plotrad_or_grid] = topoplot(all_ch_inval_1000(:, k), chanlocs,'noplot','on');
            topo = flipud(plotrad_or_grid);
            all_ch_tr_inval_1000(i, :, :, k) = topo;
        end
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
            trial_1 = mean(trial_bc(SOA200_1));
            trial_2 = mean(trial_bc(SOA200_2));
            trial_3 = mean(trial_bc(SOA200_3));
            trial_4 = mean(trial_bc(SOA200_4));
            trial_5 = mean(trial_bc(SOA200_5));
            trial_6 = mean(trial_bc(SOA200_6));
            trial_7 = mean(trial_bc(SOA200_7));
            trial_8 = mean(trial_bc(SOA200_8));
            trial_9 = mean(trial_bc(SOA200_9));
            trial_10 = mean(trial_bc(SOA200_10));
            
            % all_ch
            all_ch_val_200(j, 1) = trial_1;
            all_ch_val_200(j, 2) = trial_2;
            all_ch_val_200(j, 3) = trial_3;
            all_ch_val_200(j, 4) = trial_4;
            all_ch_val_200(j, 5) = trial_5;
            all_ch_val_200(j, 6) = trial_6;
            all_ch_val_200(j, 7) = trial_7;
            all_ch_val_200(j, 8) = trial_8;
            all_ch_val_200(j, 9) = trial_9;
            all_ch_val_200(j, 10) = trial_10;
                       
        end
        
        for k = 1:10
            [grid_or_val, plotrad_or_grid] = topoplot(all_ch_val_200(:, k), chanlocs,'noplot','on');
            topo = flipud(plotrad_or_grid);
            all_ch_tr_val_200(i, :, :, k) = topo;
        end
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
            trial_1 = mean(trial_bc(SOA200_1));
            trial_2 = mean(trial_bc(SOA200_2));
            trial_3 = mean(trial_bc(SOA200_3));
            trial_4 = mean(trial_bc(SOA200_4));
            trial_5 = mean(trial_bc(SOA200_5));
            trial_6 = mean(trial_bc(SOA200_6));
            trial_7 = mean(trial_bc(SOA200_7));
            trial_8 = mean(trial_bc(SOA200_8));
            trial_9 = mean(trial_bc(SOA200_9));
            trial_10 = mean(trial_bc(SOA200_10));
            
            % all_ch
            all_ch_inval_200(j, 1) = trial_1;
            all_ch_inval_200(j, 2) = trial_2;
            all_ch_inval_200(j, 3) = trial_3;
            all_ch_inval_200(j, 4) = trial_4;
            all_ch_inval_200(j, 5) = trial_5;
            all_ch_inval_200(j, 6) = trial_6;
            all_ch_inval_200(j, 7) = trial_7;
            all_ch_inval_200(j, 8) = trial_8;
            all_ch_inval_200(j, 9) = trial_9;
            all_ch_inval_200(j, 10) = trial_10;
                       
        end
        
        for k = 1:10
            [grid_or_val, plotrad_or_grid] = topoplot(all_ch_inval_200(:, k), chanlocs,'noplot','on');
            topo = flipud(plotrad_or_grid);
            all_ch_tr_inval_200(i, :, :, k) = topo;
        end
    end

    %% Save file '.mat'
    %m = '.mat';
    %name = strcat(foldername, m);
    %cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_Posner_paradigm\ERSP\CON_ERSP'
    %cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_Posner_paradigm\ERSP\RBD_ERSP'
    %cd 'E:\ERSP\CON_ERSP'
    %cd 'E:\ERSP\RBD_ERSP'
    
    RBD = cat(1, all_ch_tr_val_1000, all_ch_tr_inval_1000, all_ch_tr_val_200, all_ch_tr_inval_200);
    
    
    all_sub_CON = cat(1, all_sub_CON, RBD);
    %all_sub_RBD = cat(1, all_sub_RBD, RBD);
    
    
end

save('CON_topo3', 'all_sub_CON', '-v7.3')
%save('RBD_topo3', 'all_sub_RBD', '-v7.3')

toc