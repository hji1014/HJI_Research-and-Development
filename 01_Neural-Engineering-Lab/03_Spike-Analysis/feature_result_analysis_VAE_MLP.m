%%
clc;clear;

%%
% {'Easy1_0.05','Easy1_0.1','Easy1_0.15','Easy1_0.2','Easy1_0.25','Easy1_0.3','Easy1_0.35','Easy1_0.4'}
% {'Difficult1_0.05','Difficult1_0.1','Difficult1_0.15','Difficult1_0.2'}

vae_input = squeeze(spike.waveform{1, 1});
vae_input = permute(vae_input, [2 1]);
save('vae_input_Difficult1_0.2.mat', 'vae_input', '-v7.3')

%% Easy1
mean_latent = readNPY('C:/Users/Nelab_001/Documents/MATLAB/KIST_rat_monkey/spike_sorting_simulation_data_0830/hji_analysis/VAE_MLP/Easy1/Easy1_0.05/mean_model_output.npy');
var_latent = readNPY('C:/Users/Nelab_001/Documents/MATLAB/KIST_rat_monkey/spike_sorting_simulation_data_0830/hji_analysis/VAE_MLP/Easy1/Easy1_0.05/var_model_output.npy');
z_latent = readNPY('C:/Users/Nelab_001/Documents/MATLAB/KIST_rat_monkey/spike_sorting_simulation_data_0830/hji_analysis/VAE_MLP/Easy1/Easy1_0.05/sample_model_output.npy');
fake_latent = readNPY('C:/Users/Nelab_001/Documents/MATLAB/KIST_rat_monkey/spike_sorting_simulation_data_0830/hji_analysis/VAE_MLP/Easy1/Easy1_0.05/fake_model_output.npy');
fake_latent2 = readNPY('C:/Users/Nelab_001/Documents/MATLAB/KIST_rat_monkey/spike_sorting_simulation_data_0830/hji_analysis/VAE_MLP/Easy1/Easy1_0.05/fake_model2_output.npy');

outputVAE_inputDifficult14 = z_latent;
save('outputVAE_inputDifficult14.mat', 'outputVAE_inputDifficult14')

%% Difficult1
mean_latent = readNPY('C:/Users/Nelab_001/Documents/MATLAB/KIST_rat_monkey/spike_sorting_simulation_data_0830/hji_analysis/VAE_MLP/Difficult1/Difficult1_0.2/mean_model_output.npy');
var_latent = readNPY('C:/Users/Nelab_001/Documents/MATLAB/KIST_rat_monkey/spike_sorting_simulation_data_0830/hji_analysis/VAE_MLP/Difficult1/Difficult1_0.2/var_model_output.npy');
z_latent = readNPY('C:/Users/Nelab_001/Documents/MATLAB/KIST_rat_monkey/spike_sorting_simulation_data_0830/hji_analysis/VAE_MLP/Difficult1/Difficult1_0.2/sample_model_output.npy');
fake_latent = readNPY('C:/Users/Nelab_001/Documents/MATLAB/KIST_rat_monkey/spike_sorting_simulation_data_0830/hji_analysis/VAE_MLP/Difficult1/Difficult1_0.2/fake_model_output.npy');
fake_latent2 = readNPY('C:/Users/Nelab_001/Documents/MATLAB/KIST_rat_monkey/spike_sorting_simulation_data_0830/hji_analysis/VAE_MLP/Difficult1/Difficult1_0.2/fake_model2_output.npy');

outputVAE_inputDifficult14 = z_latent;
save('outputVAE_inputDifficult14.mat', 'outputVAE_inputDifficult14')

%%
for i = 1:10
    figure;
    plot(vae_input(i, :))
    hold on
    %plot(fake_latent(i, :))
    plot(fake_latent2(i, :))
    hold off
    %legend('original','generated', 'generated2')
    legend('original', 'generated')
end

figure;
plot(mean_latent(sample_num, :))
figure;
plot(var_latent(sample_num, :))
figure;
plot(z_latent(sample_num, :))