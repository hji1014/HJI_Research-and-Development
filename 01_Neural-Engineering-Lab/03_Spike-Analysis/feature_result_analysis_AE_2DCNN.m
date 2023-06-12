% figure('position', [0, 0, 150, 150]);
% present_spike = spike.waveform{1,1}(:, :, 2);
% plot(present_spike)
% axis off
% saveas(gcf,'z.jpg')
% 
% figure;
% img = imread('z.jpg');
% img_gray = rgb2gray(img);
% imshow(img_gray)
% 
% close all

%% sorting된 spike 먼저 불러오기...

spike_img = [];
for i=1:size(spike.waveform{1,1}, 3)
    present_spike = spike.waveform{1,1}(:, :, i);
    figure('position', [0, 0, 150, 150]);
    %figure('position', [680, 558, 28, 28]);
    plot(present_spike)
    axis off
    saveas(gcf,'z.jpg')
    close all
    present_img = imread('z.jpg');
    present_img_resize = imresize(present_img, [72 72]);
    present_img_gray = rgb2gray(present_img_resize);
    %present_img_gray = rgb2gray(present_img);
    spike_img(i, :, :) = present_img_gray;
end

save('spike_img.mat', 'spike_img')

%% 복원 결과 확인

extracted_output = readNPY('C:\Users\Nelab_001\Documents\MATLAB\KIST_rat_monkey\spike_sorting_simulation_data_0830\hji_analysis\AE_2DCNN\Easy1\Easy1_0.05\extracted_output.npy');
full_output = readNPY('C:\Users\Nelab_001\Documents\MATLAB\KIST_rat_monkey\spike_sorting_simulation_data_0830\hji_analysis\AE_2DCNN\Easy1\Easy1_0.05\full_output.npy');

num = 1;
ori = squeeze(spike_img(num, :, :));
fake = squeeze(full_output(num, :, :));
figure('position', [300, 300, 900, 400]);
subplot(1, 2, 1)
imagesc(ori)
colormap('gray')
subplot(1, 2, 2)
imagesc(fake)
colormap('gray')

%% 입력 값 0 or 1로 변경

for i = 1:(size(spike_img, 1)*size(spike_img, 2)*size(spike_img, 3))
    if spike_img(i) < 250
        spike_img(i) = 0;
    end
end

save('spike_img_0or255.mat', 'spike_img')

%%

output2DCNNAE_inputDifficult14 = readNPY('C:\Users\Nelab_001\Documents\MATLAB\KIST_rat_monkey\spike_sorting_simulation_data_0830\hji_analysis\AE_2DCNN\Difficult1\Difficult1_0.2\extracted_output.npy');
save('output2DCNNAE_inputDifficult14.mat', 'output2DCNNAE_inputDifficult14')