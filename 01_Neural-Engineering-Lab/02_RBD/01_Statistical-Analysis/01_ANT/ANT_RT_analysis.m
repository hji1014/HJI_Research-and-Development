clc; close all; clear;

%% 조건 별 rt 결합(input, ground_truth) -> path, save name 설정

path = 'C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\rbd\5.Incongruent';
List = dir(path);
subject_num = length(List);

subjectwise_mean = [];

for folder_num = 3:subject_num
    subject_name = List(folder_num).name;
    file_address = sprintf('./%s', subject_name);
    data = importdata(file_address);
    RT = data.RT_data;
    RT_mean = mean(RT, 1);
        
    subjectwise_mean = cat(1, subjectwise_mean, RT_mean);
end

% save('rbd_05.mat', 'input', 'ground_truth', '-v7.3')
% 
% std graph
% x = 1:10:100;
% y = [20 30 45 40 60 65 80 75 95 90];
% yneg = [0 0 0 0 0 0 0 0 0 0];
% ypos = [2 5 3 5 2 5 2 2 5 5];
% errorbar(x,y,yneg,ypos,'o')

%%

%con
con_NC = importdata('E:\ANT_RT\subjectwise_mean\control\01_NC\RT_mean.mat');
con_CC = importdata('E:\ANT_RT\subjectwise_mean\control\02_CC\RT_mean.mat');
con_SC = importdata('E:\ANT_RT\subjectwise_mean\control\03_SC\RT_mean.mat');
con_CG = importdata('E:\ANT_RT\subjectwise_mean\control\04_CG\RT_mean.mat');
con_ICG = importdata('E:\ANT_RT\subjectwise_mean\control\05_ICG\RT_mean.mat');
%rbd
rbd_NC = importdata('E:\ANT_RT\subjectwise_mean\rbd\01_NC\RT_mean.mat');
rbd_CC = importdata('E:\ANT_RT\subjectwise_mean\rbd\02_CC\RT_mean.mat');
rbd_SC = importdata('E:\ANT_RT\subjectwise_mean\rbd\03_SC\RT_mean.mat');
rbd_CG = importdata('E:\ANT_RT\subjectwise_mean\rbd\04_CG\RT_mean.mat');
rbd_ICG = importdata('E:\ANT_RT\subjectwise_mean\rbd\05_ICG\RT_mean.mat');

% figure(1)
% for i = 1:14
%     subplot(3, 5, i)
%     bar(
% end

figure(1)
x = categorical({'No cue','Center cue'});
x = reordercats(x,{'No cue','Center cue'});
subplot(2, 2, 1)
bar([con_NC con_CC])
subplot(2, 2, 2)
bar(x, [mean(con_CC, 1), mean(con_SC, 1)])
subplot(2, 2, 3)
bar([rbd_NC rbd_CC])
subplot(2, 2, 4)
bar(x, [mean(rbd_CC, 1), mean(rbd_SC, 1)])

figure(2)
subplot(1, 2, 1)
bar([mean(con_NC, 1), mean(con_CC, 1)])
subplot(1, 2, 2)
bar([mean(rbd_NC, 1), mean(rbd_CC, 1)])

figure(3)
subplot(1, 2, 1)
bar([mean(con_CC, 1), mean(con_SC, 1)])
subplot(1, 2, 2)
bar([mean(rbd_CC, 1), mean(rbd_SC, 1)])

figure(4)
subplot(1, 2, 1)
bar([mean(con_ICG, 1), mean(con_CG, 1)])
subplot(1, 2, 2)
bar([mean(rbd_ICG, 1), mean(rbd_CG, 1)])