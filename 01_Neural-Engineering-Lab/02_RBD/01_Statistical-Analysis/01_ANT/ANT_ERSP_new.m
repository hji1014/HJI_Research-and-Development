%% Initialize
clc; close all; clear;
%% 모든 피험자 Filtering, BC 이전 ERP 합치기
currentFolder = pwd;
List = dir(currentFolder);
subject_num = length(List);

all_cat = [];

for folder_num = 3:subject_num
    foldername = List(folder_num).name;
    file_address = sprintf('./%s', foldername);
    EEG_data = importdata(file_address).EEG_data;
    
    all_cat = cat(3, all_cat, EEG_data);
end

%save('D:\ANT_original_ERP_cat\rbd_ic_cat.mat', 'all_cat', '-v7.3')

%% ERSP
clc; close all; clear;

all_cat_1 = importdata('D:\ANT_original_ERP_cat\con_nc_cat.mat');
all_cat_2 = importdata('D:\ANT_original_ERP_cat\con_cc_cat.mat');

Fs = 400;
Ts = 1/Fs;
tn = linspace(-400, 2200, Fs*2.6).';
F_upper_bound = 100;
num = 5;


%% 연습

ch = 45;
trial = 500;

a = all_cat_1(ch, :, trial);
a_y = cwt_cmor_norm_var_cyc(a, Ts);
a_y_abs = abs(a_y);

b = all_cat_1(ch, :, trial);
b_y = cwt_cmor_norm_var_cycd(b, Ts, F_upper_bound, num);
b_y_abs = abs(b_y);
Iblur1 = imgaussfilt(b_y_abs,2);

figure(1)
subplot(3, 1, 1)
imagesc(flipud(a_y_abs));
colormap('jet')
colorbar
subplot(3, 1, 2)
imagesc(flipud(b_y_abs));
colormap('jet')
colorbar
subplot(3, 1, 3)
imagesc(flipud(Iblur1));
colormap('jet')
colorbar