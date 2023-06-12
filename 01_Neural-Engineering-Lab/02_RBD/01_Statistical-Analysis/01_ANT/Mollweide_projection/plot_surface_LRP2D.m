
%% 0. Configuration
%%%%%%%%%%%%%%%%%%%%%%%%%CONFIGURATION%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
base_path = pwd;
model_name = 'C3D_classic_sigmoid_ReLU';
lrp_method = 'LRP_Composite_gamma';
result_path = [base_path filesep 'Results' filesep 'dl_output_source_ERP_V1c49_trwise' filesep '200_350',...
                filesep model_name filesep lrp_method filesep];
template_path = 'G:\DL_library\template';

prefix = 'relevance';
suffix = '.npy';

mode = 'TL'; % 'Kfold', 'TL'

switch mode
    case 'TL'
        cvfolds = 50:98;
    case 'Kfold'
        cvfolds = 1:5;
end

% template_path = '/media/nelab/Hyun/DL_library/template';
template_path = 'G:\DL_library\template';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1. Load dataset

if ~exist([result_path filesep 'heatmap.mat'], 'file')
    f = waitbar(0,'Please wait...');
    heatmap_TP = [];heatmap_TPo = [];heatmap_TPors = [];outlier_ratio = [];
    for ci = cvfolds
        waitbar(1-((cvfolds(end)-ci)/length(cvfolds)),f,'Loading...');
        cvlabel = [num2str(ci) 'fold'];
        analysis_TP = readNPY([result_path filesep prefix 'TP_' cvlabel suffix]);
        [analysis_TPo, TF, L, U] = filloutliers(analysis_TP,'clip','median',1,'ThresholdFactor',3);
        outliers = isoutlier(analysis_TP,'median',1,'ThresholdFactor',3);
%         if contains(lrp_method, 'GuidedGradCam')
%             [analysis_TPo, TF, L, U] = filloutliers(analysis_TP,'clip','median',1,'ThresholdFactor',3);
%             outliers = isoutlier(analysis_TP,'median',1,'ThresholdFactor',3);
% %             [analysis_TPo, TF, L, U] = filloutliers(analysis_TP,'clip','mean',1,'ThresholdFactor',3);
% %             outliers = isoutlier(analysis_TP,'mean',1,'ThresholdFactor',3);
%         else
%             [analysis_TPo, TF, L, U] = filloutliers(analysis_TP,'clip','median',1,'ThresholdFactor',10);
%             outliers = isoutlier(analysis_TP,'median',1,'ThresholdFactor',10);
%         end
        ratio = numel(find(outliers==1))/length(outliers(:));
        disp(['outlier rate: ' num2str(ratio*100) '%, upper limit: ' num2str(round(max(U(:))*1000)/1000)]);
        outlier_ratio = cat(1, outlier_ratio, ratio);
        heatmap_TP = cat(3, heatmap_TP, squeeze(mean(analysis_TP)));
        analysis_TPo_avg = squeeze(mean(analysis_TPo));
        heatmap_TPo = cat(3, heatmap_TPo, analysis_TPo_avg);
        heatmap_TPors = cat(3, heatmap_TPors, rescale(analysis_TPo_avg, -1, 1));
    end
    close(f);
    save([result_path filesep 'heatmap.mat'], 'heatmap_TP*', 'outlier_ratio');
else
    load([result_path filesep 'heatmap.mat'], 'heatmap_TP*', 'outlier_ratio');
end

%% 2. Plot images

heatmap_avg = double(squeeze(nanmean(heatmap_TPors,3)));
% heatmap_avg = heatmap_TP(:,:,1);

% set threshold
alpha = 0.1; % 10%
heatmap_avg_sort = sortrows(heatmap_avg(:), 'descend');
thd = heatmap_avg_sort(round(length(heatmap_avg_sort)*alpha));
% heatmap_avg(heatmap_avg<thd) = 0;

cfg = [];
cfg.cmin = thd;
cfg.cmax = max(heatmap_avg(:));
cfg.summary = true;
cfg.plot = true;
sourcevalues = func_plot_surface_LRP2D(cfg, heatmap_avg);

% LRP_composite: 0 10
% LRP_composite_gamma: 0 1000

%% 3. Feature Importance Analysis
load([template_path filesep 'Mindboggle.mat']);
load([template_path filesep 'Desikan_Killiany.mat']);

load([template_path filesep 'cortex_template.mat']);
P_scs = Surface.Vertices;
P_mni = cs_convert(sMRI, 'scs', 'mni', P_scs);       % SCS   => MNI coordinates

cfg = [];
cfg.atlas = Mindboggle;
cfg.max_len = 40;
cfg.mni = P_mni; % mni coordinate
label = func_identify_brain_region(cfg, sourcevalues);
