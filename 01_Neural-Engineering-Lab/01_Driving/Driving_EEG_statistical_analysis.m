Fs = 200;                %Sampling frequency
Ts = 1/Fs;               %Sampling Period
t1 = -2:Ts:-0.005;       %Time(-2s~0s)
t2 = 0.005:Ts:5;         %Time(0s~5s)
t = -2:Ts:4.995;         %Time(-2s~5s)
length=1000;
 
o = 2^nextpow2(length);
dim = 2;
 
numfiles = 20;                                  % 피실험자 : 20명
 
% 모든 파일(피실험자) 반복문
for subject_num = 1:numfiles
    
    myfilename = sprintf('brake%d.mat', subject_num);     %일련의 파일 가져오기 (파일명+숫자.mat일 때)
    subject = importdata(myfilename);
    brake_RT = subject.brake_RT;
    brake_eeg = subject.brake_eeg;
    
    avg = mean(brake_RT);                               % avg : brake상황 평균 반응시간
    H = find(brake_RT >= avg);                          % H : 집중력 높을 때(평균 이상)
    L = find(brake_RT < avg);                           % L : 집중력 낮을 때(평균 미만)
    
    % 모든 채널 반복문
    for channel = 1:64                                  % 1~64ch 반복
        
        EEG_high = brake_eeg(channel,401:1400,H);       % brake_eeg :(앞 차 급정거 상황)EEG data / EEG_high : 집중력 높을 때의 EEG data
        EEG_low = brake_eeg(channel,401:1400,L);        % EEG_high : 집중력 낮을 때의 EEG data
        
        EEG_high_reverse = squeeze(EEG_high).';         %(집중력high)전치(행-열 바꾸기/1400 x number(H) -> number(H) x 1400)
        EEG_low_reverse = squeeze(EEG_low).';           %(집중력low)전치(행-열 바꾸기/1400 x number(L) -> number(L) x 1400)
        
        EEG_high_FFT = fft(EEG_high_reverse,o,dim);                 %집중력high 각각 FFT(number(H)번)
        high_FFT_avg = mean(abs((EEG_high_FFT)));                   %열에 대한 평균
        
        EEG_low_FFT = fft(EEG_low_reverse,o,dim);                   %집중력low  각각 FFT(number(L)번)
        low_FFT_avg = mean(abs((EEG_low_FFT)));                     %열에 대한 평균       
        
        all_ch_fft_high(channel,:) = high_FFT_avg;                  %모든 채널 값 all_ch_fft_high/low 변수에 저장
        all_ch_fft_low(channel,:) = low_FFT_avg;
        
    end
    
    all_person_fft_high(subject_num,:,:) = all_ch_fft_high(:,:);     %모든 피실험자 값 all_person_fft_high/low 변수에 저장
    all_person_fft_low(subject_num,:,:) = all_ch_fft_low(:,:);
    
end
 
% 20명, 모든 CH, 각 대역 -> 평균값 저장 // 모든 CH, 각 대역 -> 대응표본 T-TEST 후 저장
for subject_num = 1:numfiles        % 모든 피실험자 (20명)
    for channel = 1:64              % 모든 CH (64ch)
        
        person_high = all_person_fft_high(subject_num,channel,:);               %all_person_fft_high(20명,모든 channel, spectrum)
        person_high_reverse = squeeze(person_high).';                           %한원소 차원 제거
    
 
        bidirectional_spectrum1 = abs(person_high_reverse/length);        %abs : 복소수 크기or수의 절대값으로 나타내줌 / bidirectional_spectrum1 = 신호의 양방향 스펙트럼
        unidirectional_spectrum1 = bidirectional_spectrum1(:,1:o/2+1);    %unidirectional_spectrum1 : 신호의 단방향 스펙트럼
        unidirectional_spectrum1(:,2:end-1) = 2*unidirectional_spectrum1(:,2:end-1);
    
        high_x = 0:(Fs/o):(Fs/2-Fs/o);                   % 집중력 높을 때 frequency(1~100Hz)
        high_y = unidirectional_spectrum1(1,1:o/2);      % 집중력 높을 때 spectrum
        
        % avg_high -> 20(모든 피실험자) X 64(모든채널) X 5(5개 대역 평균값)
        avg_high(subject_num,channel,1) = mean2(high_y(1:22));                % 집중력 높을 때 델타파 평균
        avg_high(subject_num,channel,2) = mean2(high_y(23:41));               % 집중력 높을 때 세타파 평균
        avg_high(subject_num,channel,3) = mean2(high_y(42:67));               % 집중력 높을 때 알파파 평균
        avg_high(subject_num,channel,4) = mean2(high_y(68:159));              % 집중력 높을 때 베타파 평균
        avg_high(subject_num,channel,5) = mean2(high_y(160:257));             % 집중력 높을 때 감마파 평균
        
        person_low = all_person_fft_low(subject_num,channel,:);               %all_person_fft_low(20명,모든 channel, spectrum)
        person_low_reverse = squeeze(person_low).';                           %한원소 차원 제거  
 
        bidirectional_spectrum2 = abs(person_low_reverse/length);        %abs : 복소수 크기or수의 절대값으로 나타내줌 / bidirectional_spectrum2 = 신호의 양방향 스펙트럼
        unidirectional_spectrum2 = bidirectional_spectrum2(:,1:o/2+1);    %unidirectional_spectrum2 : 신호의 단방향 스펙트럼
        unidirectional_spectrum2(:,2:end-1) = 2*unidirectional_spectrum2(:,2:end-1);
    
        low_x = 0:(Fs/o):(Fs/2-Fs/o);                   % 집중력 낮을 때 frequency(1~100Hz)
        low_y = unidirectional_spectrum2(1,1:o/2);      % 집중력 낮을 때 spectrum
        
        % avg_high -> 20(모든 피실험자) X 64(모든채널) X 5(5개 대역 평균값)
        avg_low(subject_num,channel,1) = mean2(low_y(1:22));                % 집중력 높을 때 델타파 평균
        avg_low(subject_num,channel,2) = mean2(low_y(23:41));               % 집중력 높을 때 세타파 평균
        avg_low(subject_num,channel,3) = mean2(low_y(42:67));               % 집중력 높을 때 알파파 평균
        avg_low(subject_num,channel,4) = mean2(low_y(68:159));              % 집중력 높을 때 베타파 평균
        avg_low(subject_num,channel,5) = mean2(low_y(160:257));             % 집중력 높을 때 감마파 평균   
        
        % 대응표본 t-test (각 대역별로)
        [h_delta(channel),p_delta(channel),ci,stats{channel,1}] = ttest(squeeze(avg_high(:,channel,1)),squeeze(avg_low(:,channel,1)));
        [h_theta(channel),p_theta(channel),ci,stats{channel,2}] = ttest(squeeze(avg_high(:,channel,2)),squeeze(avg_low(:,channel,2)));
        [h_alpha(channel),p_alpha(channel),ci,stats{channel,3}] = ttest(squeeze(avg_high(:,channel,3)),squeeze(avg_low(:,channel,3)));
        [h_beta(channel),p_beta(channel),ci,stats{channel,4}] = ttest(squeeze(avg_high(:,channel,4)),squeeze(avg_low(:,channel,4)));
        [h_gamma(channel),p_gamma(channel),ci,stats{channel,5}] = ttest(squeeze(avg_high(:,channel,5)),squeeze(avg_low(:,channel,5)));
        
    end
end
 
%모든 채널 t-test(집중력 high vs low) ----> 'all_channel_t_test.mat'으로 저장
final_h = [h_delta;h_theta;h_alpha;h_beta;h_gamma];
final_p = [p_delta;p_theta;p_alpha;p_beta;p_gamma];
 
%원하는 Channel (모든 피실험자 평균) 각 대역 스펙트럼 평균 값 비교 (bar graph)
 
CH = 1;    %*********************원하는 CH 설정*****************************
 
bar_high = [mean(avg_high(:,CH,1)) mean(avg_high(:,CH,2)) mean(avg_high(:,CH,3)) mean(avg_high(:,CH,4)) mean(avg_high(:,CH,5))]; 
bar_low = [mean(avg_low(:,CH,1)) mean(avg_low(:,CH,2)) mean(avg_low(:,CH,3)) mean(avg_low(:,CH,4)) mean(avg_low(:,CH,5))];
 
bar([1 2 3 4 5],[bar_high' bar_low'])
xlabel('[EEG band]');ylabel('[Spectrum power]');
legend({'집중력 High','집중력 Low'},'Location','northeast');
ax = gca;
ax.XTick = [1 2 3 4 5];
ax.XTickLabels = {'delta','theta','alpha','beta','gamma'};
title('20명 평균 집중력 높/낮 시 각 대역별 스펙트럼');
 
% Topoplot(t-value)
 
load('chanlocs.mat')
 
%피험자 평균 델타채널 (1x64x1)
high_delta = mean(avg_high(:,:,1),1);
low_delta = mean(avg_low(:,:,1),1);
for ch =1 :64
    t_value_delta(ch) = stats{ch,1}.tstat;
end
 
[h_delta, crit_p_delta, adj_ci_cvrg_delta, adj_p_delta] = fdr_bh(p_delta);
 
figure(1)
subplot(131)
topoplot(high_delta,chanlocs,'maplimits',[0,0.7]);colorbar;
subplot(132)
topoplot(low_delta,chanlocs,'maplimits',[0,0.7]);colorbar;
subplot(133)
topoplot(t_value_delta,chanlocs,'emarker2',{find(h_delta),'o','w',10},'maplimits',[0,0.5]);colorbar;
 
%피험자 평균 세타채널 (1x64x1)
high_theta = mean(avg_high(:,:,2),1);
low_theta = mean(avg_low(:,:,2),1);
for ch =1 :64
    t_value_theta(ch) = stats{ch,2}.tstat;
end
 
[h_theta, crit_p_theta, adj_ci_cvrg_theta, adj_p_theta] = fdr_bh(p_theta);
 
figure(2)
subplot(131)
topoplot(high_theta,chanlocs,'maplimits',[0,0.7]);colorbar;
subplot(132)
topoplot(low_theta,chanlocs,'maplimits',[0,0.7]);colorbar;
subplot(133)
topoplot(t_value_theta,chanlocs,'emarker2',{find(h_theta),'o','w',10},'maplimits',[0,3.5]);colorbar;
 
%피험자 평균 알파채널 (1x64x1)
high_alpha = mean(avg_high(:,:,3),1);
low_alpha = mean(avg_low(:,:,3),1);
for ch =1 :64
    t_value_alpha(ch) = stats{ch,3}.tstat;
end
 
[h_alpha, crit_p_alpha, adj_ci_cvrg_alpha, adj_p_alpha] = fdr_bh(p_alpha);
 
figure(3)
subplot(131)
topoplot(high_alpha,chanlocs,'maplimits',[0,0.5]);colorbar;
subplot(132)
topoplot(low_alpha,chanlocs,'maplimits',[0,0.5]);colorbar;
subplot(133)
topoplot(t_value_alpha,chanlocs,'emarker2',{find(h_alpha),'o','w',10},'maplimits',[0,3.5]);colorbar;    %알아서 조절해서 tp1,tp2 비교되게끔 만들기
 
%피험자 평균 베타채널 (1x64x1)
high_beta = mean(avg_high(:,:,4),1);
low_beta = mean(avg_low(:,:,4),1);
for ch =1 :64
    t_value_beta(ch) = stats{ch,4}.tstat;
end
 
[h_beta, crit_p_beta, adj_ci_cvrg_beta, adj_p_beta] = fdr_bh(p_beta,0.08);              %유의수준 0.08(약간 덜 엄격)
 
figure(4)
subplot(131)
topoplot(high_beta,chanlocs,'maplimits',[0,0.3]);colorbar;
subplot(132)
topoplot(low_beta,chanlocs,'maplimits',[0,0.3]);colorbar;
subplot(133)
topoplot(t_value_beta,chanlocs,'emarker2',{find(h_beta),'o','w',10},'maplimits',[0,3]);colorbar;
 
%피험자 평균 감마채널 (1x64x1)
high_gamma = mean(avg_high(:,:,5),1);
low_gamma = mean(avg_low(:,:,5),1);
for ch =1 :64
    t_value_gamma(ch) = stats{ch,5}.tstat;
end
 
[h_gamma, crit_p_gamma, adj_ci_cvrg_gamma, adj_p_gamma] = fdr_bh(p_gamma);
 
figure(5)
subplot(131)
topoplot(high_gamma,chanlocs,'maplimits',[0,0.2]);colorbar;
subplot(132)
topoplot(low_gamma,chanlocs,'maplimits',[0,0.2]);colorbar;
subplot(133)
topoplot(t_value_gamma,chanlocs,'emarker2',{find(h_gamma),'o','w',10},'maplimits',[0,2]);colorbar;
 
%그룹화 한 채널(앞,뒤 : 총 2개) 피실험자 각각(20명) 스펙트럼 평균내기(alpha band, beta band)
alpha_region_front = [1 2 4 5 6 7 10 33 34 35 36 37 38 39 40 46];   %alpha 대역 유의한 채널(앞)
alpha_region_behind = [20 24 25 26 53 57 58 59 62 63 64];           %alpha 대역 유의한 채널(뒤)
beta_region_front = [1 3 7 8 33 36 40 41 42 45];                    %beta 대역 유의한 채널(앞)
beta_region_behind = [23 24 25 26 51 54 56 57 58 59 60 62 63 64];   %beta 대역 유의한 채널(뒤)
 
for subject_num = 1:20
    all_alpha_region_front_high(subject_num) = mean(avg_high(subject_num,alpha_region_front,3));      %(20명모두,집중high) alpha_region_front 채널의 평균 (1 X 20)
    all_alpha_region_behind_high(subject_num) = mean(avg_high(subject_num,alpha_region_behind,3));    %(20명모두,집중high) alpha_region_behind 채널의 평균 (1 X 20)
    all_alpha_region_front_low(subject_num) = mean(avg_low(subject_num,alpha_region_front,3));        %(20명모두,집중low) alpha_region_front 채널의 평균 (1 X 20)
    all_alpha_region_behind_low(subject_num) = mean(avg_low(subject_num,alpha_region_behind,3));      %(20명모두,집중low) alpha_region_behind 채널의 평균 (1 X 20)
    
    all_beta_region_front_high(subject_num) = mean(avg_high(subject_num,beta_region_front,4));        %(20명모두,집중high) beta_region_front 채널의 평균 (1 X 20)
    all_beta_region_behind_high(subject_num) = mean(avg_high(subject_num,beta_region_behind,4));      %(20명모두,집중high) beta_region_behind 채널의 평균 (1 X 20)
    all_beta_region_front_low(subject_num) = mean(avg_low(subject_num,beta_region_front,4));          %(20명모두,집중low) beta_region_front 채널의 평균 (1 X 20)
    all_beta_region_behind_low(subject_num) = mean(avg_low(subject_num,beta_region_behind,4));        %(20명모두,집중low) beta_region_behind 채널의 평균 (1 X 20)
end
 
%앞, 뒷 t-test //데이터 추출 방법 : stats_alpha_front.tstat;     //각각 높/낮 평균 스펙트럼
%bar그래프로 표현
%그렇게 하여 실제 사용할 수 있는 그림을 만들기 (글자 크기, 색깔 세부적으로 신경써서)
[h_alpha_front,p_alpha_front,ci_alpha_front,stats_alpha_front] = ttest(all_alpha_region_front_high,all_alpha_region_front_low);         %alpha_region_front 집중(높/낮) 대응표본 t-test
[h_alpha_behind,p_alpha_behind,ci_alpha_behind,stats_alpha_behind] = ttest(all_alpha_region_behind_high,all_alpha_region_behind_low);   %alpha_region_behind 집중(높/낮) 대응표본 t-test
[h_beta_front,p_beta_front,ci_beta_front,stats_beta_front] = ttest(all_beta_region_front_high,all_beta_region_front_low);               %beta_region_front 집중(높/낮) 대응표본 t-test
[h_beta_behind,p_beta_behind,ci_beta_behind,stats_beta_behind] = ttest(all_beta_region_behind_high,all_beta_region_behind_low);         %beta_region_behind 집중(높/낮) 대응표본 t-test
 
%전두 alpha-band (집중 높/낮) 주파수 스펙트럼
figure(1)
bar_alpha_front_high_low = [mean(all_alpha_region_front_high) mean(all_alpha_region_front_low)];        %alpha front 집중 높/낮 20명평균 스펙트럼
hold on;
bar(1,mean(all_alpha_region_front_high),0.5);
bar(1.7,mean(all_alpha_region_front_low),0.5);
%title('[집중도에 따른 전두 α-band 스펙트럼]','fontsize',20);
%xlabel('(집중도)','fontsize',10);
ylabel('(Spectrum power)','fontsize',20);ylim([0 0.5]);
ax = gca;
ax.XTick = [1 1.7];
ax.XTickLabels = {'집중력 high','집중력 low'};
 
err_alpha_front_high = std(all_alpha_region_front_high)/sqrt(20);   %에러바  / 표준오차
err_alpha_front_low = std(all_alpha_region_front_low)/sqrt(20);
err1 = errorbar(1,mean(all_alpha_region_front_high),err_alpha_front_high,'.','Color','black');
err2 = errorbar(1.7,mean(all_alpha_region_front_low),err_alpha_front_low,'.','Color','black');
err1.LineWidth = 1.3;
err2.LineWidth = 1.3;
%legend({'집중력 high','집중력 low'},'Location','northeast');
hold off;
 
%두정,후두 alpha-band (집중 높/낮) 주파수 스펙트럼
figure(2)
bar_alpha_behind_high_low = [mean(all_alpha_region_behind_high) mean(all_alpha_region_behind_low)];        %alpha behind 집중 높/낮 20명평균 스펙트럼
hold on;
bar(1,mean(all_alpha_region_behind_high),0.5);
bar(1.7,mean(all_alpha_region_behind_low),0.5);
%title('[집중도에 따른 두정,후두 α-band 스펙트럼]','fontsize',15);
%xlabel('(집중도)','fontsize',10);
ylabel('(Spectrum power)','fontsize',20);ylim([0 0.6]);
ax = gca;
ax.XTick = [1 1.7];
ax.XTickLabels = {'집중력 high','집중력 low'};
 
err_alpha_behind_high = std(all_alpha_region_behind_high)/sqrt(20);   %에러바
err_alpha_behind_low = std(all_alpha_region_behind_low)/sqrt(20);
err1 = errorbar(1,mean(all_alpha_region_behind_high),err_alpha_behind_high,'.','Color','black');
err2 = errorbar(1.7,mean(all_alpha_region_behind_low),err_alpha_behind_low,'.','Color','black');
err1.LineWidth = 1.3;
err2.LineWidth = 1.3;
%legend({'집중력 high','집중력 low'},'Location','northeast');
hold off;
 
%전두 beta-band (집중 높/낮) 주파수 스펙트럼
figure(3)
bar_beta_front_high_low = [mean(all_beta_region_front_high) mean(all_beta_region_front_low)];        %alpha front 집중 높/낮 20명평균 스펙트럼
hold on;
bar(1,mean(all_beta_region_front_high),0.5);
bar(1.7,mean(all_beta_region_front_low),0.5);
%title('[집중도에 따른 전두 β-band 스펙트럼]','fontsize',15);
%xlabel('(집중도)','fontsize',10);
ylabel('(Spectrum power)','fontsize',20);ylim([0 0.22]);
ax = gca;
ax.XTick = [1 1.7];
ax.XTickLabels = {'집중력 high','집중력 low'};
 
err_beta_front_high = std(all_beta_region_front_high)/sqrt(20);   %에러바  / 표준오차
err_beta_front_low = std(all_beta_region_front_low)/sqrt(20);
err1 = errorbar(1,mean(all_beta_region_front_high),err_beta_front_high,'.','Color','black');
err2 = errorbar(1.7,mean(all_beta_region_front_low),err_beta_front_low,'.','Color','black');
err1.LineWidth = 1.3;
err2.LineWidth = 1.3;
%legend({'집중력 high','집중력 low'},'Location','northeast');
hold off;
 
%두정,후두부 beta-band (집중 높/낮) 주파수 스펙트럼
figure(4)
bar_beta_behind_high_low = [mean(all_beta_region_behind_high) mean(all_beta_region_behind_low)];        %alpha front 집중 높/낮 20명평균 스펙트럼
hold on;
bar(1,mean(all_beta_region_behind_high),0.5);
bar(1.7,mean(all_beta_region_behind_low),0.5);
%title('[집중도에 따른 두정,후두부 β-band 스펙트럼]','fontsize',15);
%xlabel('(집중도)','fontsize',10);
ylabel('(Spectrum power)','fontsize',20);ylim([0 0.27]);
ax = gca;
ax.XTick = [1 1.7];
ax.XTickLabels = {'집중력 high','집중력 low'};
 
err_beta_behind_high = std(all_beta_region_behind_high)/sqrt(20);   %에러바  / 표준오차
err_beta_behind_low = std(all_beta_region_behind_low)/sqrt(20);
err1 = errorbar(1,mean(all_beta_region_behind_high),err_beta_behind_high,'.','Color','black');
err2 = errorbar(1.7,mean(all_beta_region_behind_low),err_beta_behind_low,'.','Color','black');
err1.LineWidth = 1.3;
err2.LineWidth = 1.3;
%legend({'집중력 high','집중력 low'},'Location','northeast');
hold off;
 
% 작업공간 변수를 파일에 저장
%save('all_person_EEG_FFT','all_person_fft_high','all_person_fft_low');
%save('all_channel_t_test','final_h','final_p');
