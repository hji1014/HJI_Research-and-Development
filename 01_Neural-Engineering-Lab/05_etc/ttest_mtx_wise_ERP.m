function [t_observed,P_observed]=ttest_mtx_wise_ERP(ERP_mtx)

% ttest_mtx_wise는 ttest를 매트릭스 수준에서 할 수 있게 해주는 함수임.
% ERP_mtx는 ERP 신호를 담고 있는 matrix임.
% ERP_mtx는 WHO*EVENT*ELECTRODES * TIME의 순서로 배열을 만들어서 넣어주길 바람.
% t_observed는 계산된 t 값들이 나오게 됨.
% P_observed는 계산된 t 값에 대응되는 p-value가 나오게 됨.

% 참고문헌 : 일반통계학 / 영지문화사 / 김우철 외 공저

[n_who,n_event,n_electrodes,n_time]=size(ERP_mtx);

dataA=ERP_mtx(:,1,:,:);
dataB=ERP_mtx(:,2,:,:);

diff=dataA-dataB; % 두 데이터의 차이를 구함.

%% difference in sample means
diff_smpl_mns=sum(diff,1)/n_who; % 두 데이터의 차이의 평균

%% standard deviations of each sample group
% 두 데이터의 표준 오차 계산 (평균의 표준편차 계산과 같은 것임)
diff_smpl_var=sum((diff-repmat(diff_smpl_mns,n_who,1)).^2)/(n_who-1);

%% evaluate t and p
t_observed=squeeze(diff_smpl_mns./sqrt(diff_smpl_var/n_who));
P_observed=1-tcdf(t_observed,n_who-1);

end