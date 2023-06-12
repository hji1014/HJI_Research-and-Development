function y=cwt_cmor_norm_var_cycd(data,Ts,F_upper_bound, num)

% y=cwt_cmor_norm_var_cyc(data,Ts)�� ���� ���·� �Է���.
% Ts�� 1/fs�� ����
% num : ���ļ� ������ ������ ���� �ֱ�
% 

TF_temp=zeros(F_upper_bound,length(data));
cycles=linspace(8,25,F_upper_bound/num);
for i=1:length(cycles)
    fb(i)=(cycles(i)/pi/sqrt(2))^(1/2);
    wname=['cmor',num2str(fb(i)),'-1'];
    scale=[num*(i-1)+1:num*i];
    fact=centfrq(wname)/Ts;
    scal=fact*(1./scale);
    TF_temp(num*(i-1)+1:num*i,:)=cwt(data,scal,wname);
end
y=TF_temp;
