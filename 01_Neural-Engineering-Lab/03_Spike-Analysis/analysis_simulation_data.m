%%
clc;clear;

do_import = false;
do_spike_sorting = true;
do_explore = false;

%% 0. Configuration
save_path = 'Analysis_sim';
N_TRIAL = 1; % 1:8
FDIM = 'tsne'; % 'pca', 'tsne'
TRIALTYPE = 'Easy1'; % 'Easy1', 'Difficult1'
DET = 'STEO'; % 'amp', 'TEO', 'STEO'

% import data
if do_import
    %% 1. Load data
    cfg = [];
    cfg.label = {'Simulated'};
    %cfg.path = 'G:\data_analysis_ex\spike_sorting_simulation_data\Simulator';
    cfg.path = 'C:\Users\Nelab_001\Documents\MATLAB\KIST_rat_monkey\spike_sorting_simulation_data_0830\Simulator';
    cfg.trialtype = 'Difficult1';
    Data = mat2fieldtrip(cfg);
%     % noise level estimation
%     cfg = [];
%     cfg.method = 'median';
%     Data = ftc_noise_estimation(cfg, Data);
    
    %save([save_path filesep 'raw_' TRIALTYPE '.mat'], 'Data');
end

if do_spike_sorting
    load([save_path filesep 'raw_' TRIALTYPE '.mat'], 'Data');
    
    %% 1. bandpass filtering
    % 알고리즘 검증이 목적이므로 simulated data에서 본 과정 생략.
%     참고문헌에서도 raw waveform에서 바로 spike detection 수행하였음 (Quiroga et al., 2004)
%     cfg = [];
%     cfg.bpfilter = 'yes';
%     cfg.bpfreq = [500 3000];
%     data_bpf = ft_preprocessing(cfg, Data);

%     cfg = [];
%     cfg.method = 'square';
%     Data = ftc_signal_operator(cfg, Data);
    
    %% 2. automatic thresholding
    cfg = [];
    cfg.method = 'median';
    Data = ftc_noise_estimation(cfg, Data);
    
    %% 3. spike detection
    cfg = [];
%     cfg.threshold = 0.3; % use manual threshold
    cfg.threshold = 'STEO'; % 'amp' 'TEO' 'STEO'
    cfg.alignment = 'yes';
    cfg.trial = N_TRIAL;
    spike = ftc_spike_detection(cfg, Data);
    
%     % alignment
%     cfg             = [];
%     cfg.fsample     = data_bpf.fsample;
%     cfg.rejectonpeak = 'no';
%     cfg.interpolate = 1; % keep the density of samples as is
%     [wave, spikeCleaned] = ft_spike_waveform(cfg,spike);
    
%     % plotting spike
%     figure;plot(squeeze(spike.waveform{1,1}));
    
    %% 4. spike sorting
    c_num = 3; % number of clusters
    
    cfg = [];
    cfg.fext = 'waveform';
    cfg.fdim = FDIM;
    cfg.clustering = 'K-means';
    cfg.clustering_fnum = 2;
    cfg.clustering_K = c_num;
    cfg.plot = true;
    spike_s = ftc_spike_sorting(cfg, spike);
    
    if cfg.plot
        % plotting spikes
        c_wav = cell(1,c_num);
        for ci = 1:c_num
            c_wav{ci} = transpose(squeeze(spike_s.waveform{ci}));
        end
        
        figure;plot(mean(c_wav{1}), 'r');hold on;plot(mean(c_wav{2}), 'b');hold on;plot(mean(c_wav{3}), 'g');
        title('Averaged spike signal');
        legend('Cluster 1','Cluster 2','Cluster 3');
        xlabel('Time sample');ylabel('Amplitude');
        colors = {'r', 'b', 'g'};
        
        figure;
        for pi = 1:3
            subplot(3,1,pi);plot(c_wav{pi}', colors{pi});title(['Cluster #' num2str(pi)]);
            xlabel('Time sample');ylabel('Amplitude');
        end
    end
    
    %% 5. Evaluation
    % groundtruth
    ground_truth = [Data.cfg.spike_times{1,N_TRIAL}{1,1}; Data.cfg.spike_class{1,N_TRIAL}{1,1}];
    cluster_num = length(unique(ground_truth(2,:)));
    
    gt_timestamp = cell(1, cluster_num);
    gt_spike = cell(1, cluster_num);
    for spi = 1:size(ground_truth,2)
        c_num = ground_truth(2,spi);
        gt_timestamp{1,c_num} = cat(2, gt_timestamp{1,c_num}, ground_truth(1,spi));
        signal = Data.trial{1,N_TRIAL}(ground_truth(1,spi):ground_truth(1,spi)+63);
        gt_spike{1,c_num} = cat(1, gt_spike{1,c_num}, signal);
    end
    gt_template = zeros(cluster_num,64);
    for ci = 1:size(gt_template,1)
        gt_template(ci,:) = mean(gt_spike{1,ci});
    end
% %     % plot
%     figure;plot(gt_template(1,:), 'r');hold on;plot(gt_template(2,:), 'b');hold on;plot(gt_template(3,:), 'g');
%     title('Ground truth');
%     legend('Cluster 1','Cluster 2','Cluster 3');
%     xlabel('Time sample');ylabel('Amplitude');
    
    % evaluation
    TP = 0;
    FP = 0;
    spike_marker = zeros(1,size(ground_truth,2));
    spike_classes = cell(1,cluster_num);
    for ci = 1:cluster_num
        preds = spike_s.timestamp{1,ci};
        for ai = 1:size(preds, 2)
            [M, I] = min(abs(ground_truth(1,:)-preds(ai)));
            if M < 10 % 5: strict
                TP = TP + 1;
                spike_classes{1,ci} = cat(1, spike_classes{1,ci}, ground_truth(2,I));
                spike_marker(1,I) = spike_marker(1,I) + 1;
            else
                FP = FP + 1;
            end
        end
    end
    FN = numel(find(spike_marker==0));
    
    answer_cnum = [];
    wrong_cnum = 0; % wrong cluster number
    for ci = 1:cluster_num
        wrong_cnum = wrong_cnum + size(spike_classes{1,ci},1)-numel(find(spike_classes{1,ci}==mode(spike_classes{1,ci})));
        answer_cnum = cat(1, answer_cnum, mode(spike_classes{1,ci}));
    end
    
    SNR = 10*log10(mean(spike.peak_amps)/Data.noiseinfo{N_TRIAL}); % estimated SNR = 10*log10(S/N) where S: mean |peak amplitude|, N: estimated noise level
    type = {Data.trialinfo{N_TRIAL}};
    detection = TP+FP;
    detection_acc = (TP/(TP+FP+FN))*100;
    algorithm = {spike_s.cfg.sortinginfo};
    if numel(unique(answer_cnum)) == cluster_num
        sorting_acc = ((detection-wrong_cnum)/detection)*100;
    else
        sorting_acc = NaN;
    end 
    summary = table(type, SNR, {DET}, detection, TP, FP, FN, detection_acc, algorithm, sorting_acc);
    
    save([save_path filesep 'result.mat'], 'summary');
    
    %% 6. save results
    % clustering results
    X_pc = spike_s.pc;
    C = spike_s.cluster_centroid;
    
    h = plot(X_pc(spike_s.spike_label==1,1),X_pc(spike_s.spike_label==1,2),'r.','MarkerSize',12);hold on;
    plot(X_pc(spike_s.spike_label==2,1),X_pc(spike_s.spike_label==2,2),'b.','MarkerSize',12);hold on;
    plot(X_pc(spike_s.spike_label==3,1),X_pc(spike_s.spike_label==3,2),'g.','MarkerSize',12);hold on;
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
    saveas(h, [save_path filesep 'clustering_result_' type{1} '_' algorithm{1} '.png']);
    close(gcf);
    
    % spike template
    c_wav = cell(1,cluster_num);
    for ci = 1:cluster_num
        c_wav{ci} = transpose(squeeze(spike_s.waveform{ci}));
    end
    h = plot(mean(c_wav{1}), 'r');hold on;plot(mean(c_wav{2}), 'b');hold on;plot(mean(c_wav{3}), 'g');
    title('Averaged spike signal');
    legend('Cluster 1','Cluster 2','Cluster 3');
    xlabel('Time sample');ylabel('Amplitude');
    colors = {'r', 'b', 'g'};
    saveas(h, [save_path filesep 'spike_template_' type{1} '_' algorithm{1} '.png']);
    close(gcf);
    
end


if do_explore
    load([save_path filesep 'raw_' TRIALTYPE '.mat'], 'Data');
    
    %% plot raw waveform
    cfg = [];
    cfg.blocksize = 0.1;                             % Length of data to display, in seconds
    cfg.trial = N_TRIAL;
    ft_databrowser(cfg, Data);
    xlabel('Time (s)');ylabel('Amplitude');
    
    %% plot power spectrum
    signal_005 = Data.trial{1,1}(find(Data.time{1}==0):find(Data.time{1}==2));
    [pxx_005, f] = pwelch(signal_005,256,[],[0:6000],Data.fsample);
    signal_04 = Data.trial{1,8}(find(Data.time{1}==0):find(Data.time{1}==2));
    [pxx_04, f] = pwelch(signal_04,256,[],[0:6000],Data.fsample);
    figure;plot(f,10*log10(pxx_005), 'b');hold on;
    plot(f,10*log10(pxx_04), 'r');hold on;
    xlabel('frequency [Hz]');ylabel('amplitude [dB]');
    legend('Easy1-0.05', 'Easy1-0.4');
end
