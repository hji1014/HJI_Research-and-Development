function sourcevalues = func_plot_surface_LRP2D(cfg, heatmap)

% dimension of heatmap should be at least two
% heatmap dimension (H x W x F x N)
% H, W: feature dim.
% F: number of fold (optional)
% N: number of repetition (optional)
%
% cfg.anatomy_path: anatomy file path (necessary)
% cfg.cmin: z color (min)
% cfg.cmax: z color (max)
% cfg.surfalpha: alpha (transparaency)
% cfg.summary: true/false // print descriptives
% cfg.plot: true/false // plot surface image

if ~isfield(cfg, 'anatomy_path')
    cfg.anatomy_path = 'tess_cortex_pial_low.mat'; % Default: ICBM152
end
if ~isfield(cfg, 'cmin')
    cfg.cmin = 'min';
end
if ~isfield(cfg, 'cmax')
    cfg.cmax = 'max';
end
if ~isfield(cfg, 'surfalpha')
    cfg.surfalpha = 0; % transparency
end
if ~isfield(cfg, 'plot')
    cfg.plot = true; % transparency
end

ctx = load(cfg.anatomy_path, 'Vertices', 'Faces');


heatmap_avg = double(squeeze(mean(mean(heatmap,4),3)));
n_nan = numel(find(isnan(heatmap_avg)==1));
if n_nan
    warning(['ignored NaN values: ' num2str(n_nan)]);
    heatmap_avg = double(squeeze(nanmean(nanmean(heatmap,4),3)));
end
    
% define regions of left and right hemishpere
lH_image = squeeze(heatmap_avg(:,1:60));
rH_image = squeeze(heatmap_avg(:,61:120));

[sourcevalues] = inflate_source(lH_image, rH_image, cfg.anatomy_path);
% % visualization
% % 1. projection_vis
% figure;imagesc(rH_image);
% for ti = 1:12
%    figure(1);subplot(4,3,ti);imagesc(squeeze(heatmap_avg(ti,:,:)),[0 0.1]);
% end
if cfg.summary
    disp(['min value: ' num2str(min(sourcevalues))]);
    disp(['max value: ' num2str(max(sourcevalues))]);
    disp(['mean value: ' num2str(mean(sourcevalues))]);
end

if cfg.plot
    gray_map = colormap('jet'); close;
    
    if strcmp(cfg.cmin, 'min')
        cmin = min(sourcevalues);
    else
        cmin = cfg.cmin;
    end
    if strcmp(cfg.cmax, 'max')
        cmax = max(sourcevalues);
    else
        cmax = cfg.cmax;
    end

    Color_ref = linspace(cmin,cmax,length(gray_map));
    Surf_color = zeros(length(sourcevalues), 3);
    for i = 1:length(sourcevalues)
        [~, idx] = min(abs(Color_ref-sourcevalues(i,1)));
        Surf_color(i,:) = gray_map(idx,:);
    %     indices(i,:) = idx;
    end

    [hFig, iDS, iFig] = view_surface_matrix(ctx.Vertices, ctx.Faces, cfg.surfalpha, Surf_color);
    if isfield(cfg, 'save_path')
        saveas(hFig, cfg.save_path);
    end
end
