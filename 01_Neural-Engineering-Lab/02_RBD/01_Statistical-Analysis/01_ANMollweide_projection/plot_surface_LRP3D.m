%% 0. Configuration
%%%%%%%%%%%%%%%%%%%%%%%%%CONFIGURATION%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
base_path = pwd;
model_name = 'C3D_classic_sigmoid_ReLU';

lrp_method = 'GuidedGradCam';
result_path = [base_path filesep 'Results' filesep 'dl_output_source_ERP_V1c49_trwise' filesep 'whole',...
                filesep model_name filesep lrp_method filesep];
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
%% 0. Outlier detection
ci = 50;
cvlabel = [num2str(ci) 'fold'];
analysis_TP = readNPY([result_path filesep prefix 'TP_' cvlabel suffix]);

trial_n = 67;
sample = squeeze(analysis_TP(trial_n,:,:,:));
max(sample, [], [1, 2]);
test = sample(:,:,11);
figure;imagesc(test, [0 100]);
cfg = [];
cfg.cmin = 0;
cfg.cmax = 1;
cfg.summary = false;
cfg.plot = true;
sourcevalues = func_plot_surface_LRP2D(cfg, test);
analysis_TP_cut = analysis_TP;
for ti = 1:size(analysis_TP, 1)
    temp = squeeze(analysis_TP(ti,:,:,:));
    thd = mean(temp, 'all')+std(sample, [], 'all')*10;
    temp(temp>thd) = 0;
    analysis_TP_cut(ti,:,:,:) = temp;
end
trial_max = max(analysis_TP_cut, [], [2, 3, 4]);
% 
% for i = 1:49
% imagesc(heatmap_TP(:,:,12,i));pause(0.3);
% end

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
%         if strcmp(lrp_method, 'GuidedGradCam')
%             [analysis_TPo, TF, L, U] = filloutliers(analysis_TP,'clip','mean',1,'ThresholdFactor',3);
%             outliers = isoutlier(analysis_TP,'mean',1,'ThresholdFactor',3);
%         else
%             [analysis_TPo, TF, L, U] = filloutliers(analysis_TP,'clip','median',1,'ThresholdFactor',10);
%             outliers = isoutlier(analysis_TP,'median',1,'ThresholdFactor',10);
%         end
        ratio = numel(find(outliers==1))/length(outliers(:));
        disp(['outlier rate: ' num2str(ratio*100) '%, upper limit: ' num2str(round(max(U(:))*1000)/1000)]);
        outlier_ratio = cat(1, outlier_ratio, ratio);
%         for ti = 1:size(analysis_TP, 1)
%             test = squeeze(analysis_TP(ti,:,:,:));
%             tempo = filloutliers(test(:),'clip','median','ThresholdFactor',1000);
%             tempo = reshape(tempo, 120, 120, 16);
%         end
%         figure;
%         for ti = 1:16
%             subplot(4,4,ti);imagesc(test(:,:,ti), [min(test(:)) max(test(:))]);
%         end
%         zero_rate = 0;
%         for ti = 1:size(analysis_TP, 1)
%             test = squeeze(analysis_TP(ti,:,:,:));
%             if sum(test, 'all') ~= 0
%                 zero_rate = zero_rate + 1;
%             end
%         end
        heatmap_TP = cat(4, heatmap_TP, squeeze(mean(analysis_TP)));
        analysis_TPo_avg = squeeze(mean(analysis_TPo));
        heatmap_TPo = cat(4, heatmap_TPo, analysis_TPo_avg);
        heatmap_TPors = cat(4, heatmap_TPors, rescale(analysis_TPo_avg, -1, 1));
    end
    close(f);
    save([result_path filesep 'heatmap.mat'], 'heatmap_TP*', 'outlier_ratio');
else
    load([result_path filesep 'heatmap.mat'], 'heatmap_TP*', 'outlier_ratio');
end


%% 2. Plot images
% heatmap_TPsz = heatmap_TPz;
% for t1 = 1:size(heatmap_TPsz,1)
%     for t2 = 1:size(heatmap_TPsz,2)
%         for t3 = 1:size(heatmap_TPsz,3)
%             heatmap_TPsz(t1,t2,t3,:) = (heatmap_TPsz(t1,t2,t3,:)-nanmean(squeeze(heatmap_TPsz(t1,t2,t3,:))))./nanstd(squeeze(heatmap_TPsz(t1,t2,t3,:)));
%         end
%     end
% end

heatmap_avg = double(squeeze(nanmean(nanmean(heatmap_TP,5),4)));
epoch = 50;
% heatmap_avg = abs(heatmap_avg);

% set threshold
alpha = 0.001; % Top N*100 %, ReLU: 0.01 for GGC, 0.0001 for LRP_CMP; LeakyReLU: 0.01 for GGC, 0.0001 for LRP_CMP
heatmap_avg_sort = sortrows(heatmap_avg(:), 'descend');
thd = heatmap_avg_sort(round(length(heatmap_avg_sort)*alpha));
% thd = 0; % for LRP_Composite

cfg = [];
cfg.cmin = 0;
cfg.cmax = 0.00001;
% cfg.cmin = -10;
% cfg.cmax = 10;
% cfg.cmin = 0;
% cfg.cmax = 6000;
% cfg.cmin = 0.05;
% cfg.cmax = 0.15;
% cfg.cmin = min(heatmap_avg(:));
cfg.cmax = 0;
cfg.summary = true;
cfg.plot = false; % '2D', '3D', false

cfg.cmin = thd;
cfg.cmax = max(heatmap_avg(:));
cfg.summary = false;
cfg.plot = '3D'; % '2D', '3D', false
% set figure
[~, rFig] = func_plot_surface_LRP3D(cfg, heatmap_avg(:,:,1));
cfg.hFig = rFig;
% figure;
val_max = [];
sourcevalues = [];
for ti = 1:16
%     subplot(4,4,ti);
    disp(['time: ' num2str(epoch*(ti-1)) '_' num2str(epoch*ti)]);
    cfg.save_path = [result_path filesep 'images' filesep num2str(ti) '_avg_test_' num2str(epoch*(ti-1)) '_' num2str(epoch*ti) '.png'];
    [sourcevalue, ~, stat] = func_plot_surface_LRP3D(cfg, heatmap_avg(:,:,ti));
    val_max = cat(1, val_max, stat.max);
    sourcevalues = cat(2, sourcevalues, sourcevalue);
%     title(['time: ' num2str(epoch*(ti-1)) '_' num2str(epoch*ti)]);
end
save([result_path filesep 'sourcevalues.mat'], 'sourcevalues');

%% 2.2. Plot images within critical time periods
heatmap_avg = mean(heatmap_avg(:,:,5:7), 3);

% set threshold
alpha = 0.1; % 0.1 for GGC
heatmap_avg_sort = sortrows(heatmap_avg(:), 'descend');
thd = heatmap_avg_sort(round(length(heatmap_avg_sort)*alpha));
% heatmap_avg(heatmap_avg<thd) = 0;

cfg = [];
cfg.cmin = thd;
cfg.cmax = max(heatmap_avg(:));
cfg.summary = true;
cfg.plot = true;
sourcevalue = func_plot_surface_LRP2D(cfg, heatmap_avg);

%% 3. Feature Importance Analysis
load([template_path filesep 'Mindboggle.mat']);
load([template_path filesep 'Desikan_Killiany.mat']);

lrp_method = 'LRP_Composite_gamma';
result_path = [base_path filesep 'Results' filesep 'dl_output_source_ERP_V1c49_trwise' filesep 'whole',...
                filesep model_name filesep lrp_method filesep];
load([result_path filesep 'sourcevalues.mat'], 'sourcevalues');

load([template_path filesep 'cortex_template.mat']);
P_scs = Surface.Vertices;
P_mni = cs_convert(sMRI, 'scs', 'mni', P_scs);       % SCS   => MNI coordinates

switch lrp_method
    case 'GuidedGradCam'
        cfg = [];
        cfg.atlas = Mindboggle;
        cfg.max_len = 40;
        cfg.mni = P_mni; % mni coordinate
        label = func_identify_brain_region(cfg, sourcevalues);
        lLO_vertices = cell2mat(label(strcmp(label(:,1), 'lateraloccipital L'),2));
        rLO_vertices = cell2mat(label(strcmp(label(:,1), 'lateraloccipital R'),2));
        rSP_vertices = cell2mat(label(strcmp(label(:,1), 'superiorparietal R'),2));
%         label = func_identify_brain_region(cfg, sourcevalues(:,7));
%         lLO_vertices = cell2mat(label(1:4,2));
%         rLO_vertices = cell2mat(label(5:14,2));
%         rSP_vertices = cell2mat(label(15:16,2));
    case 'LRP_Composite_gamma'
        cfg = [];
        cfg.atlas = Mindboggle;
        cfg.max_len = 40;
        cfg.mni = P_mni; % mni coordinate
        label = func_identify_brain_region(cfg, sourcevalues);
        lLO_vertices = cell2mat(label(strcmp(label(:,1), 'lateraloccipital L'),2));
        rLO_vertices = cell2mat(label(strcmp(label(:,1), 'lateraloccipital R'),2));
        rSP_vertices = cell2mat(label(strcmp(label(:,1), 'superiorparietal R'),2));
%         label = func_identify_brain_region(cfg, sourcevalues(:,7));
%         lLO_vertices = cell2mat(label(1:3,2));
%         rLO_vertices = cell2mat(label(8:23,2));
%         rSP_vertices = cell2mat(label(24:28,2));
        
end

save([result_path filesep 'ROI.mat'], '*_vertices');

% % get MNI coordinates 

% load('cortex_template.mat');
% P_scs = Surface.Vertices(idx,:);
% P_mni = cs_convert(sMRI, 'scs', 'mni', P_scs);       % SCS   => MNI coordinates

%% 4. Plot Time series for ROI
% load([template_path filesep 'Mindboggle.mat']);
% lLO_vertices = Mindboggle(19).Vertices;
% rLO_vertices = Mindboggle(20).Vertices;
% rSP_vertices = Mindboggle(56).Vertices;

lrp_method = 'GuidedGradCam';
result_path = [base_path filesep 'Results' filesep 'dl_output_source_ERP_V1c49_trwise' filesep 'whole',...
                filesep model_name filesep lrp_method filesep];
load([result_path filesep 'sourcevalues.mat'], 'sourcevalues');
load([result_path filesep 'ROI.mat'], '*_vertices');
lLO_GGC = mean(sourcevalues(unique(lLO_vertices),:),1);
rLO_GGC = mean(sourcevalues(unique(rLO_vertices),:),1);
rSP_GGC = mean(sourcevalues(unique(rSP_vertices),:),1);

lrp_method = 'LRP_Composite_gamma';
result_path = [base_path filesep 'Results' filesep 'dl_output_source_ERP_V1c49_trwise' filesep 'whole',...
                filesep model_name filesep lrp_method filesep];
load([result_path filesep 'sourcevalues.mat'], 'sourcevalues');
load([result_path filesep 'ROI.mat'], '*_vertices');
lLO_LRP = mean(sourcevalues(unique(lLO_vertices),:),1);
rLO_LRP = mean(sourcevalues(unique(rLO_vertices),:),1);
rSP_LRP = mean(sourcevalues(unique(rSP_vertices),:),1);


figure;subplot(3,1,1);h1 = plot(lLO_GGC, '--gs','MarkerSize',10,'MarkerEdgeColor','b','MarkerFaceColor','w');hold on;
[M, I] = max(lLO_GGC);plot(I, M, 's','MarkerSize',10,'MarkerEdgeColor','b','MarkerFaceColor','r');hold on;
h2 = plot(lLO_LRP, '--ms','MarkerSize',10,'MarkerEdgeColor','b','MarkerFaceColor','w');hold on;
[M, I] = max(lLO_LRP);plot(I, M, 's','MarkerSize',10,'MarkerEdgeColor','b','MarkerFaceColor','r');hold on;
ylabel('lLO');set(gca, 'XTick',[], 'YTick', [], 'fontsize', 18);
legend([h1, h2], {'GGCAM','LRP'});

subplot(3,1,2);plot(rLO_GGC, '--gs','MarkerSize',10,'MarkerEdgeColor','b','MarkerFaceColor','w');hold on;
[M, I] = max(rLO_GGC);plot(I, M, 's','MarkerSize',10,'MarkerEdgeColor','b','MarkerFaceColor','r');hold on;
plot(rLO_LRP, '--ms','MarkerSize',10,'MarkerEdgeColor','b','MarkerFaceColor','w');hold on;
[M, I] = max(rLO_LRP);plot(I, M, 's','MarkerSize',10,'MarkerEdgeColor','b','MarkerFaceColor','r');hold on;
ylabel('rLO');set(gca, 'XTick',[], 'YTick', [], 'fontsize', 18);

subplot(3,1,3);plot(rSP_GGC, '--gs','MarkerSize',10,'MarkerEdgeColor','b','MarkerFaceColor','w');hold on;
[M, I] = max(rSP_GGC);plot(I, M, 's','MarkerSize',10,'MarkerEdgeColor','b','MarkerFaceColor','r');hold on;
plot(rSP_LRP, '--ms','MarkerSize',10,'MarkerEdgeColor','b','MarkerFaceColor','w');hold on;
[M, I] = max(rSP_LRP);plot(I, M, 's','MarkerSize',10,'MarkerEdgeColor','b','MarkerFaceColor','r');hold on;
ylabel('rSP');set(gca, 'YTick', [], 'fontsize', 18);xticks(1:2:16);xticklabels(0:100:750);


