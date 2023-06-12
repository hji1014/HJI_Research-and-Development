%% 주의사항
% 1. path 설정

%% Initialize
clc; close all; clear;

%% Averaging of all subject(Grand AVG)

%path =  'C:\Users\Nelab_001\Documents\MATLAB\RBD_Posner_paradigm\control_post\good';
%path =  'C:\Users\Nelab_001\Documents\MATLAB\RBD_Posner_paradigm\control_post\good+bad';
path =  'C:\Users\Nelab_001\Documents\MATLAB\RBD_Posner_paradigm\RBD_post\good';
%path =  'C:\Users\Nelab_001\Documents\MATLAB\RBD_Posner_paradigm\RBD_post\good+bad';

List = dir(path);
subject_num = length(List);

% avg 장소 설정
valid_200_chavg_v = zeros(60, 800);
invalid_200_chavg_v = zeros(60, 800);
valid_1000_chavg_v = zeros(60, 1200);
invalid_1000_chavg_v = zeros(60, 1200);

for folder_num = 3:subject_num
    
    foldername = List(folder_num).name;
    s = load(foldername);
    
    valid_200_chavg_v = valid_200_chavg_v + s.valid_200_chavg;
    invalid_200_chavg_v = invalid_200_chavg_v + s.invalid_200_chavg;
    valid_1000_chavg_v = valid_1000_chavg_v + s.valid_1000_chavg;
    invalid_1000_chavg_v = invalid_1000_chavg_v + s.invalid_1000_chavg;
    
end

valid_200_chavg_v_complete = valid_200_chavg_v/(subject_num-2);
invalid_200_chavg_v_complete = invalid_200_chavg_v/(subject_num-2);
valid_1000_chavg_v_complete = valid_1000_chavg_v/(subject_num-2);
invalid_1000_chavg_v_complete = invalid_1000_chavg_v/(subject_num-2);

save('result', 'valid_200_chavg_v_complete', 'invalid_200_chavg_v_complete', 'valid_1000_chavg_v_complete','invalid_1000_chavg_v_complete')

%d = [1, 2, 3, 4, 5];
%c = [0];
%for i = 1:5
%    c = c+d(i);
%end
