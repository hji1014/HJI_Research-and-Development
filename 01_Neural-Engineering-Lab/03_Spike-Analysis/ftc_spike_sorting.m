function spike2 = ftc_spike_sorting(cfg, spike)
% spike sorting
% feature extraction & dimension reduction & clustering
% input data: spike structure
% cfg.fext: 'waveform' % set feature extraction method
% cfg.fdim: 'pca', 'tsne' % set dimension reduction method
% cfg.clustering: 'K-means' % set clustering method
% cfg.clustering_fnum: set number of feature dimensions as an input of clustering
% cfg.clustering_K: set number of clusters to cluster
% cfg.plot: true, false
% cfg.groundtruth_template: spike signal [cluster x 64]; match cluster number with templates of ground_truth of spike signal

rng(1); % for reproducibility

if ~strcmp(spike.label{1}, 'unsorted')
    error('An unsorted spike data is available only.');
end
if ~isfield(cfg, 'fext')
    cfg.fext = 'waveform';
end
if ~isfield(cfg, 'fdim')
    cfg.fdim = 'tsne';
end
if ~isfield(cfg, 'clustering')
    cfg.clustering = 'K-means';
end
if ~isfield(cfg, 'fdim_num')
    cfg.clustering_fnum = 2;
end
if ~isfield(cfg, 'clustering_K')
    cfg.clustering_K = 3;
end
if ~isfield(cfg, 'plot')
    cfg.plot = false;
end

spike2 = spike;
spike2.cfg = struct;
spike2.cfg.featureinfo = cfg.fext;
spike2.cfg.sortinginfo = [cfg.fdim '_' cfg.clustering];

%% 1. Feature extraction
switch cfg.fext
    case 'waveform'
        X = transpose(squeeze(spike.waveform{1}));
end

%% 2. Feature dimension reduction
switch cfg.fdim
    case 'pca'
        [coeff,score,~,~,explained] = pca(X);
    case 'tsne'
        score = tsne(X);
end

%% 3. Clustering
X_pc = score(:,1:cfg.clustering_fnum);
switch cfg.clustering
    case 'K-means'
        opts = statset('Display','final');
        [idx,C] = kmeans(X_pc,cfg.clustering_K,'Distance','cityblock',...
            'Replicates',5,'Options',opts);
end

for ci = 1:cfg.clustering_K
    spike2.label{1,ci} = ['Cluster ' num2str(ci)];
    spike2.timestamp{1,ci} = spike.timestamp{1}(idx==ci);
    spike2.waveform{1,ci} = spike.waveform{1}(:,:,idx==ci);
end
spike2.pc = X_pc;
spike2.cluster_centroid = C;
spike2.spike_label = idx;



%% Visualization
if cfg.plot
    % plotting feature
    figure;plot(score(:,1), score(:,2), 'k.', 'MarkerSize', 12);
    title('feature dimension reduction result');
    if strcmp(cfg.fdim, 'pca')
        xlabel('PC 1');ylabel('PC 2');
    else
        xlabel('Dimension 1');ylabel('Dimension 2');
    end
    
    % plotting cluster
    figure;
    plot(X_pc(idx==1,1),X_pc(idx==1,2),'r.','MarkerSize',12);hold on;
    plot(X_pc(idx==2,1),X_pc(idx==2,2),'b.','MarkerSize',12);hold on;
    plot(X_pc(idx==3,1),X_pc(idx==3,2),'g.','MarkerSize',12);hold on;
    plot(C(:,1),C(:,2),'kx',...
         'MarkerSize',15,'LineWidth',3) 
    legend('Cluster 1','Cluster 2','Cluster 3','Centroids',...
           'Location','NW')
    title('Cluster Assignments and Centroids');
    hold off
    if strcmp(cfg.fdim, 'pca')
        xlabel('PC 1');ylabel('PC 2');
    else
        xlabel('Dimension 1');ylabel('Dimension 2');
    end
    
end