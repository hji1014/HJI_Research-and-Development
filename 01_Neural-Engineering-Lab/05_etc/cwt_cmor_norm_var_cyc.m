function y=cwt_cmor_norm_var_cyc(data2,Ts)

TF_temp=zeros(100,length(data2));
cycles=[4:0.25:8.75];
for i=1:10
    fb(i)=(cycles(i)/pi/sqrt(2))^2;
    wname=['cmor',num2str(fb(i)),'-1'];
   
    scale=[1*(i-1)+0.5:0.5:5*i];
    fact=centfrq(wname)/Ts;
    scal=fact*(1./scale);
    TF_temp(2*(i-1)+1:10*i,:)=cwt(data2,scal,wname);
end
y=TF_temp;