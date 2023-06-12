function spike = ftc_spike_detection(cfg, data)
% spike detection
% input data: bandpass-filtered signals
% data.noiseinfo field is necessary
% method 참고: doi 10.1162/089976604774201631 (Quiroga, 2004)
% cfg.trial: only work for single trial
% cfg.threshold: 'amp': amplitude thresholding, 'TEO', 'STEO', 
% 'plot_all': for validation
% cfg.direction: 'positive', 'negative', 'both'
% cfg.plot: true, false

if ~isfield(cfg, 'trial')
    error('define trial');
end
if ~isfield(cfg, 'threshold')
    cfg.threshold = 'amp';
end
if ~isfield(cfg, 'threshold_alpha')
    switch cfg.threshold
        case 'amp'
            cfg.threshold_multiplier = 4;
        otherwise % TEO
            cfg.threshold_multiplier = 8;
    end
end
if ~isfield(cfg, 'alignment')
    cfg.alignment = 'yes';
end
if ~isfield(cfg, 'plot')
    cfg.plot = false;
end


if ~isfield(cfg, 'direction')
    cfg.direction = 'both'; % # TODO
end

spike = struct;
spike.dimord = '{chan}_lead_time_spike';
spike.label = {'unsorted'};
spike.timestamp = cell(1,1);
spike.waveform = cell(1,1);
Fs = data.fsample;
ti = cfg.trial;

switch cfg.direction
    case 'negative'
        timeseries = -data.trial{1,ti};
        error('not applicapable # TODO');
    otherwise
        timeseries = data.trial{1,ti};
end

% amplitude thresholding
cfg2 = [];
cfg2.method = 'median';
data = ftc_noise_estimation(cfg2, data);
data.threshold.amplitude = cfg.threshold_multiplier*data.noiseinfo{1,ti};
% TEO
signal = [NaN timeseries NaN];
TEO = signal(2:end-1).^2-signal(3:end).*signal(1:end-2);
data.threshold.TEO = cfg.threshold_multiplier*nanmean(TEO);

% STEO
win = hamming(5)';
STEO = conv(TEO, win);
STEO = STEO(3:end-2);
data.threshold.STEO = cfg.threshold_multiplier*nanmean(STEO);

% ts = 6000;
% figure;
% plot(data.time{1,ti}(1:ts), data.trial{1,ti}(1:ts), 'k:');hold on;
% plot(data.time{1,ti}(1:ts), TEO(1:ts), 'b--');
% plot(data.time{1,ti}(1:ts), STEO(1:ts), 'r--');
% xlabel('Time (s)');ylabel('Amplitude');legend('Original signal', 'TEO', 'STEO');
% set(gca, 'fontsize', 18);
% figure;
% plot(data.time{1,ti}(1:ts), STEO(1:ts), 'b');hold on;
% plot([data.time{1,ti}(1) data.time{1,ti}(ts)], [data.threshold.STEO data.threshold.STEO], 'r--');
% xlabel('Time (s)');ylabel('Amplitude');title('STEO');
% set(gca, 'fontsize', 18);

if ischar(cfg.threshold)
    if strcmp(cfg.threshold, 'amp')
        signal_est = timeseries;
        threshold = data.threshold.amplitude;
    else
        switch cfg.threshold
            case 'TEO'
                signal_est = TEO;
                threshold = data.threshold.TEO;
            case 'STEO'
                signal_est = STEO;
                threshold = data.threshold.STEO;
        end
    end
else
    threshold = cfg.threshold; % manual thresholding
end
% if cfg.plot
%     figure;plot(timeseries(1:1000)
% end

% peak detection
% [~, locs] = findpeaks(signal_est, Fs, 'MinPeakHeight', threshold, 'MinPeakDistance', 0.0005);
% [~, locs_sample] = findpeaks(signal_est, 'MinPeakHeight', threshold, 'MinPeakDistance', 0.0005*Fs);
[~, locs_sample] = findpeaks(signal_est, 'MinPeakHeight', threshold, 'MinPeakDistance', 0.001*Fs);
% findpeaks(signal_est(1:5000), 'MinPeakHeight', threshold, 'MinPeakDistance', 0.001*Fs)
% findpeaks(signal_est(102700:103000), 'MinPeakHeight', threshold, 'MinPeakDistance', 0.001*Fs)
% plotting
if (strcmp(cfg.plot, 'yes')) && (ti == cfg.plot_trial)
    figure;plot(data.time{1,ti}(1:end), signal_est(1:end));hold on;
    plot([data.time{1,ti}(1) data.time{1,ti}(end)], [threshold threshold], 'r--');hold on;
    plot(data.time{1,ti}(1:end), timeseries(1:end), 'g--');
    % when error
%     search_space = locs_sample(spi)-50:locs_sample(spi)+50;
% %     figure;plot(signal_est(search_space));hold on;
%     figure;plot(timeseries(search_space), 'g--');hold on;
%     findpeaks(signal_est(search_space), 'MinPeakHeight', threshold, 'MinPeakDistance', 0.001*Fs)
%     legend('Original signal', 'STEO');
%         % chunk data
%         figure;findpeaks(timeseries(45000:46000), Fs, 'MinPeakHeight', threshold, 'MinPeakDistance', 0.0005);
% %         figure;findpeaks(timeseries(45000:46000), Fs, 'MinPeakHeight', threshold, 'MinPeakDistance', 0.002);
%         hold on;plot([0 Data.time{1,ti}(46000)-Data.time{1,ti}(45000)], [threshold, threshold], 'r--');
end

% extracting spikes (64 samples)
% locs_sample = round(locs*Fs);

spike.peak_amps = zeros(1,length(locs_sample));
spike.waveform{1,1} = zeros(length(data.label), 64, length(locs_sample));
spike.timestamp{1,1} = zeros(1,length(locs_sample));
% if strcmp(cfg.alignment, 'yes') % DEPRECATED
%     spike.waveform{1,1} = zeros(length(data.label), 256, length(locs_sample));
% end
for spi = 1:length(locs_sample)
    if (locs_sample(spi)-20 < 0) || (locs_sample(spi)+43 > length(timeseries))
        continue;
    end
%     plot(timeseries(locs_sample(spi)-50:locs_sample(spi)+50), 'g--');
    waveform = timeseries(locs_sample(spi)-20:locs_sample(spi)+43);
%     TEO_peak = STEO(locs_sample(spi)-20:locs_sample(spi)+43);
    if strcmp(cfg.alignment, 'yes')
        x = linspace(0,256,64);
        xx = linspace(0,256,256);
        waveform_spline = csapi(x, waveform, xx);
%         search_time_stamp = 40;
        search_time_stamp = 79;
        [ppks, plocs] = findpeaks(waveform_spline(80-search_time_stamp:80+search_time_stamp), 'SortStr', 'descend');
        [npks, nlocs] = findpeaks(-waveform_spline(80-search_time_stamp:80+search_time_stamp), 'SortStr', 'descend');
        if isempty(plocs)
            ppks = NaN; plocs = 999;
        end
        if isempty(nlocs)
            npks = NaN; nlocs = 999;
        end
        pks = [ppks(1), npks(1)];
        locs = [plocs(1), nlocs(1)];
        [~, I] = min(abs(80-locs));
        spike.peak_amps(1,spi) = pks(I); loc = locs(I);
        loc = loc+(80-search_time_stamp);
        waveform_align = [];
        for li = loc-80:loc+175
            if li<1
                waveform_align = cat(2, waveform_align, 0);
            else
                try 
                    waveform_align = cat(2, waveform_align, waveform_spline(li));
                catch 
                    waveform_align = cat(2, waveform_align, 0);
                end
            end
        end
        if numel(find(waveform_align==0))>100
            error('abnormal signal found. check it');
        end
        waveform = downsample(waveform_align,4); % downsampling to original sampling rate
    end
    spike.waveform{1,1}(1,:,spi) = waveform;
    spike.timestamp{1,1}(spi) = locs_sample(spi)-20; % spike onset
end
