%% 주의사항
% 1. path 설정

%% Initialize
clc; close all; clear;

%% Brain region

frontal = 10:14;
central = 28:32;
parietal = 46:50;
parieto_occipital = [44, 45, 46, 50, 51, 52, 53, 54, 56, 57];

%% Averaging of all subject(Grand AVG)

%path =  'C:\Users\Nelab_001\Documents\MATLAB\RBD_Posner_paradigm\control_post\good';
%path =  'C:\Users\Nelab_001\Documents\MATLAB\RBD_Posner_paradigm\control_post\good+bad';
path =  'C:\Users\Nelab_001\Documents\MATLAB\RBD_Posner_paradigm\RBD_post\good';
%path =  'C:\Users\Nelab_001\Documents\MATLAB\RBD_Posner_paradigm\RBD_post\good+bad';

List = dir(path);
subject_num = length(List);

% N1 장소 설정
% frontal
valid_200_N1_frontal = zeros(subject_num-2, 1);
invalid_200_N1_frontal = zeros(subject_num-2, 1);
valid_1000_N1_frontal = zeros(subject_num-2, 1);
invalid_1000_N1_frontal = zeros(subject_num-2, 1);
% central
valid_200_N1_central = zeros(subject_num-2, 1);
invalid_200_N1_central = zeros(subject_num-2, 1);
valid_1000_N1_central = zeros(subject_num-2, 1);
invalid_1000_N1_central = zeros(subject_num-2, 1);
% parietal
valid_200_N1_parietal = zeros(subject_num-2, 1);
invalid_200_N1_parietal = zeros(subject_num-2, 1);
valid_1000_N1_parietal = zeros(subject_num-2, 1);
invalid_1000_N1_parietal = zeros(subject_num-2, 1);
% parieto_occipital
valid_200_N1_parieto_occipital = zeros(subject_num-2, 1);
invalid_200_N1_parieto_occipital = zeros(subject_num-2, 1);
valid_1000_N1_parieto_occipital = zeros(subject_num-2, 1);
invalid_1000_N1_parieto_occipital = zeros(subject_num-2, 1);


for folder_num = 3:subject_num
    
    foldername = List(folder_num).name;
    s = load(foldername);
    
    %frontal N1
    valid_200_chavg_N1_frontal = mean(s.valid_200_chavg_N1(frontal, :), 1);
    invalid_200_chavg_N1_frontal = mean(s.invalid_200_chavg_N1(frontal, :), 1);
    valid_1000_chavg_N1_frontal = mean(s.valid_1000_chavg_N1(frontal, :), 1);
    invalid_1000_chavg_N1_frontal = mean(s.invalid_1000_chavg_N1(frontal, :), 1);
    
    valid_200_N1_frontal(folder_num-2, 1) = valid_200_chavg_N1_frontal;
    invalid_200_N1_frontal(folder_num-2, 1) = invalid_200_chavg_N1_frontal;
    valid_1000_N1_frontal(folder_num-2, 1) = valid_1000_chavg_N1_frontal;
    invalid_1000_N1_frontal(folder_num-2, 1) = invalid_1000_chavg_N1_frontal;
    
    %central N1
    valid_200_chavg_N1_central = mean(s.valid_200_chavg_N1(central, :), 1);
    invalid_200_chavg_N1_central = mean(s.invalid_200_chavg_N1(central, :), 1);
    valid_1000_chavg_N1_central = mean(s.valid_1000_chavg_N1(central, :), 1);
    invalid_1000_chavg_N1_central = mean(s.invalid_1000_chavg_N1(central, :), 1);
    
    valid_200_N1_central(folder_num-2, 1) = valid_200_chavg_N1_central;
    invalid_200_N1_central(folder_num-2, 1) = invalid_200_chavg_N1_central;
    valid_1000_N1_central(folder_num-2, 1) = valid_1000_chavg_N1_central;
    invalid_1000_N1_central(folder_num-2, 1) = invalid_1000_chavg_N1_central;
    
    %parietal N1
    valid_200_chavg_N1_parietal = mean(s.valid_200_chavg_N1(parietal, :), 1);
    invalid_200_chavg_N1_parietal = mean(s.invalid_200_chavg_N1(parietal, :), 1);
    valid_1000_chavg_N1_parietal = mean(s.valid_1000_chavg_N1(parietal, :), 1);
    invalid_1000_chavg_N1_parietal = mean(s.invalid_1000_chavg_N1(parietal, :), 1);
    
    valid_200_N1_parietal(folder_num-2, 1) = valid_200_chavg_N1_parietal;
    invalid_200_N1_parietal(folder_num-2, 1) = invalid_200_chavg_N1_parietal;
    valid_1000_N1_parietal(folder_num-2, 1) = valid_1000_chavg_N1_parietal;
    invalid_1000_N1_parietal(folder_num-2, 1) = invalid_1000_chavg_N1_parietal;
    
    %parietal N1
    valid_200_chavg_N1_parieto_occipital = mean(s.valid_200_chavg_N1(parieto_occipital, :), 1);
    invalid_200_chavg_N1_parieto_occipital = mean(s.invalid_200_chavg_N1(parieto_occipital, :), 1);
    valid_1000_chavg_N1_parieto_occipital = mean(s.valid_1000_chavg_N1(parieto_occipital, :), 1);
    invalid_1000_chavg_N1_parieto_occipital = mean(s.invalid_1000_chavg_N1(parieto_occipital, :), 1);
    
    valid_200_N1_parieto_occipital(folder_num-2, 1) = valid_200_chavg_N1_parieto_occipital;
    invalid_200_N1_parieto_occipital(folder_num-2, 1) = invalid_200_chavg_N1_parieto_occipital;
    valid_1000_N1_parieto_occipital(folder_num-2, 1) = valid_1000_chavg_N1_parieto_occipital;
    invalid_1000_N1_parieto_occipital(folder_num-2, 1) = invalid_1000_chavg_N1_parieto_occipital;
    
end
