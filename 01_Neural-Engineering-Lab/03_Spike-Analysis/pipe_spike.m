type_label = {'Easy1', 'Difficult1'};
trials = [8, 4];
fdim_label = {'pca', 'tsne'};

%% Sorting algorithm (dimension reduction) 성능 비교
total_summary = table;
for tyi = 1:2
    TRIALTYPE = type_label{tyi};
    for ni = 1:trials(tyi)
        for fi = 1:2
            N_TRIAL = ni;
            FDIM = fdim_label{fi};
            eval('analysis_simulation_data');
            load(['Analysis_sim' filesep 'result.mat'], 'summary');
            total_summary = cat(1, total_summary, summary);
        end
    end
end

save(['Analysis_sim' filesep 'total_result.mat'], 'total_summary');


%% Detection 성능 비교 (Easy1, n0.05)
total_summary = table;

type_label = {'Easy1', 'Difficult1'};
trials = [8, 4];
fdim_label = {'pca', 'tsne'};
det_label = {'amp', 'TEO', 'STEO'};

TRIALTYPE = type_label{1};
for ni = 4
    for fi = 1:3
        N_TRIAL = ni;
        FDIM = fdim_label{1};
        DET = det_label{fi};
        eval('analysis_simulation_data');
        load(['Analysis_sim' filesep 'result.mat'], 'summary');
        total_summary = cat(1, total_summary, summary);
    end
end

save(['Analysis_sim' filesep 'total_result.mat'], 'total_summary');