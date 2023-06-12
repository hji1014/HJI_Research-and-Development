%% 이전 rat 코드
% stimulation onset
tri = sum(stim_data, 1)/2;
stim_idx = find(tri);
stim_onset_idx = [];
for i = 1:34
    stim_onset_idx(i) = stim_idx(1+12*(i-1));
end
stim_onset_second = stim_onset_idx/30000;       % stimulation onset time

figure(1)
plot(t, tri)

a = amplifier_data_BPF(12, :);
b = amplifier_data_BPF(12, :);
%[pks,locs] = findpeaks(a);
[pks,locs] = findpeaks(a, 'MinPeakHeight', 10);

t_1 = t( 30000*13+1:30000*14);

figure(5)
plot(t_1, a, t_1(locs), pks, 'or')
ylim([-25 25]);


tri_idx0 = find(tri);
tri_idx = find(stim_data(:, :));
amp = stim_data(12, tri);

% data visualization
for i = 1:32
%     figure(1)
%     subplot(4, 8, i)
%     plot(t, amplifier_data(i, :))
    figure(2)
    subplot(8, 4, i)
    plot(t, stim_data(i, :))
end
tri = sum(stim_data, 1);
figure(3)
plot(t, tri)


for i = 1:32
    figure(3)
    subplot(4, 8, i)
    plot(t(:, 1:30000), amplifier_data(i, 1:30000))
    figure(4)
    subplot(4, 8, i)
    plot(t(:, 1:30000), stim_data(i, 1:30000))
end

a = amplifier_data(12, :);
a_HPF = HPF(a, 30000, 250);     % HPF(in, fSample, fCutoff)
b = stim_data(12, :);

figure(1)
subplot(3, 1, 1)
plot(t, a)
subplot(3, 1, 2)
plot(t, a_HPF)
subplot(3, 1, 3)
plot(t, b)

figure(2)
subplot(3, 1, 1)
plot(t(100000:110000), a(100000:110000))
subplot(3, 1, 2)
plot(t(100000:110000), a_HPF(100000:110000))
subplot(3, 1, 3)
plot(t(100000:110000), b(100000:110000))

figure(3)
plot(t, stim_data(12, :))

figure(4)                                                   % 63900:64500 -> 2.13~2.15
subplot(2, 1, 1)
plot(t(63900:64500), a_HPF(63900:64500))
subplot(2, 1, 2)
plot(t(63900:64500), stim_data(12, 63900:64500))

figure(5)
for i=1:32
    subplot(4, 8, i)
    plot(t(63900:64500), amplifier_data_BPF(i, 63900:64500))
end

figure(6)
plot(t(63900:64500), amplifier_data_BPF(20, 63900:64500))