function [lH_image, rH_image] = flatten_source(source_path, surface_path, size_x, size_y, data_path, cfg)

% From 3D spherical representation to 2D projection source images
%
% Code to project registered source images from a spherical representation 
% to a mollweide 2D projection representation.
%
% INPUT:
%       - cfg (struct): option for details
%       - source_path (char): path to the source image .mat file
%       - surface_path (char): path to the surface file .mat file
%       - size_x (double): x length of 2D projection
%       - size_y (double): y length of 2D projection
%       - data_path (char): path to the recording file .mat file
%
% OUTPUT:
%       - lH_image (double matrix): the resulting 2D projection of the left
%       hemisphere
%       - rH_image (double matrix): the resulting 2D projection of the
%       right hemisphere
%
% FUNCTION DEPENDENCIES:
%       - tess_hemisplit: This function is part of the Brainstorm software:
%         https://neuroimage.usc.edu/brainstorm
%       - mollweide_proj: Code written by Saskia van Heumen
%
%  Written by: Saskia van Heumen, September 2019
%  Written for: Internship at Centre of Pain Medicine, Erasmus MC,
%  Rotterdam

%  Revised by Hyun Kim (dosteps@yonsei.ac.kr)
%  Revised 20.04.09: add 'ImagingKernel' option of the source image
%                    add 'size_x', 'size_y' variables for input
%                    add for loop at line 98, related with 'freq_no'
    
% Load the values of the sourceimages
sourcemap = load(source_path);

% Load the surface file containing the spherical coordinates
surface = load(surface_path);

% Load the data file of the scalp EEG if it needs (e.g. shared kernel)
if isempty(cfg)
    cfg = [];
end

if ~isempty(data_path)
    data = load(data_path);
end

% Find the indexes of the right and left hemispheres
[rH_idx, lH_idx, ~, ~, ~, ~] = tess_hemisplit(surface);

% Get the values of the source image
if isfield(sourcemap, 'ImageGridAmp') && ~isempty(sourcemap.ImageGridAmp)
    sourcevalues = sourcemap.ImageGridAmp(:, 1);
elseif isfield(sourcemap, 'TF')
    sourcevalues = squeeze(sourcemap.TF);
elseif isfield(sourcemap, 'pmap')
    sourcevalues = squeeze(sourcemap.tmap);
elseif isfield(sourcemap, 'ImagingKernel')
    kernelvalues = squeeze(sourcemap.ImagingKernel);
    if ~isempty(data_path)
        sourcevalues = kernelvalues*data.F;
    else
        error('flatten_source: Empty data_path');
    end
else
    error('The sourcemap structure must contain an ImageGridAmp, TF, tmap field or ImagingKernel');
end


% check TF configurations
if isfield(cfg,'TF')
    cfg.TF.time = sourcemap.Time;
    cfg.TF.Freqs = sourcemap.Freqs;
    sourcevalues = nanmean(nanmean(sourcevalues(:,cfg.TF.TOI(1)<=cfg.TF.time & cfg.TF.time<=cfg.TF.TOI(end),...
        cfg.TF.FOI(1)<=cfg.TF.Freqs & cfg.TF.Freqs<=cfg.TF.FOI(end)),2),3);
end

% Get spherical vertices of the right and left hemisphere
lHvertices = double(surface.Reg.Sphere.Vertices(lH_idx, :));
rHvertices = double(surface.Reg.Sphere.Vertices(rH_idx, :));

% Check if the number of vertices correspond to the number of values
if (length(lHvertices) + length(rHvertices)) ~= length(sourcevalues)
    error('Number of vertices do not correspond to the number of values')
end

% Get the latidude and longitude coordinates of the vertices
[lH_lon, lH_lat, lH_radius] = cart2sph(lHvertices(:, 1), ...
    lHvertices(:, 2), lHvertices(:, 3));
[rH_lon, rH_lat, rH_radius] = cart2sph(rHvertices(:, 1), ...
    rHvertices(:, 2), rHvertices(:, 3));

% Get the 2D projection coordinates using mollweide projection
[lH_x, lH_y] = mollweide_proj(lH_lat, lH_lon, mean(lH_radius(:)));
[rH_x, rH_y] = mollweide_proj(rH_lat, rH_lon, mean(rH_radius(:)));

% Get a equally spaced coordinate system for the image (uniform grid)
lH_xgrid = linspace(min(lH_x), max(lH_x), size_x);
lH_ygrid = linspace(min(lH_y), max(lH_y), size_y);
[lH_xnew, lH_ynew] = meshgrid(lH_xgrid, lH_ygrid);

rH_xgrid = linspace(min(rH_x), max(rH_x), size_x);
rH_ygrid = linspace(min(rH_y), max(rH_y), size_y);
[rH_xnew, rH_ynew] = meshgrid(rH_xgrid, rH_ygrid);

% Interpolate the scattered data on the uniform grid

lH_image = []; rH_image = [];
for freq_no = 1:size(sourcevalues,2)
    lH_image(:,:, freq_no) = griddata(lH_x, lH_y, sourcevalues(lH_idx, freq_no), ...
        lH_xnew, lH_ynew, 'linear');
    rH_image(:,:, freq_no) = griddata(rH_x, rH_y, sourcevalues(rH_idx, freq_no), ...
        rH_xnew, rH_ynew, 'linear');
end

end


