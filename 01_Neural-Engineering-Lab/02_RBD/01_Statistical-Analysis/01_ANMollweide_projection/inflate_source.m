function [sourcevalues] = inflate_source(lH_image, rH_image, surface_path)

% From 3D spherical representation to 2D projection source images
%
%
%  Written by Hyun Kim (dosteps@yonsei.ac.kr)
%
% Code to reconstruct the mollweide 2D projection representation
% to a registered source images from a spherical representation.
%
% INPUT:
%       - lH_image (double matrix): the resulting 2D projection of the left
%       hemisphere
%       - rH_image (double matrix): the resulting 2D projection of the
%       right hemisphere
%
% OUTPUT:
%       - sourcevalues (double matrix): the values of the source image
%
% FUNCTION DEPENDENCIES:
%       - tess_hemisplit: This function is part of the Brainstorm software:
%         https://neuroimage.usc.edu/brainstorm
%       - mollweide_proj: Code written by Saskia van Heumen
%
%
%  Original code Written by: Saskia van Heumen, September 2019
%  Written for: Internship at Centre of Pain Medicine, Erasmus MC,
%  Rotterdam

% Load the surface file containing the spherical coordinates
surface = load(surface_path);

% Find the indexes of the right and left hemispheres
[rH_idx, lH_idx, ~, ~, ~, ~] = tess_hemisplit(surface);

% Get spherical vertices of the right and left hemisphere
lHvertices = double(surface.Reg.Sphere.Vertices(lH_idx, :));
rHvertices = double(surface.Reg.Sphere.Vertices(rH_idx, :));

% Get the latidude and longitude coordinates of the vertices
[lH_lon, lH_lat, lH_radius] = cart2sph(lHvertices(:, 1), ...
    lHvertices(:, 2), lHvertices(:, 3));
[rH_lon, rH_lat, rH_radius] = cart2sph(rHvertices(:, 1), ...
    rHvertices(:, 2), rHvertices(:, 3));

% Get the 2D projection coordinates using mollweide projection
[lH_x, lH_y] = mollweide_proj(lH_lat, lH_lon, mean(lH_radius(:)));
[rH_x, rH_y] = mollweide_proj(rH_lat, rH_lon, mean(rH_radius(:)));

% Get a equally spaced coordinate system for the image (uniform grid)
size_x = size(lH_image,2); size_y = size(lH_image,1);
lH_xgrid = linspace(min(lH_x), max(lH_x), size_x);
lH_ygrid = linspace(min(lH_y), max(lH_y), size_y);
[lH_xnew, lH_ynew] = meshgrid(lH_xgrid, lH_ygrid);

rH_xgrid = linspace(min(rH_x), max(rH_x), size_x);
rH_ygrid = linspace(min(rH_y), max(rH_y), size_y);
[rH_xnew, rH_ynew] = meshgrid(rH_xgrid, rH_ygrid);

% Interpolate the projection on the scattered data
sourcevalues = zeros(length(lH_idx)+length(rH_idx),size(lH_image,3));
for freq_no = 1:size(lH_image,3)
    sourcevalues(lH_idx, freq_no) = griddata(lH_xnew, lH_ynew, squeeze(lH_image(:,:,freq_no)), ...
        lH_x, lH_y, 'linear');
    sourcevalues(rH_idx, freq_no) = griddata(rH_xnew, rH_ynew, squeeze(rH_image(:,:,freq_no)), ...
        rH_x, rH_y, 'linear');
end

end


