function [g,cut,alp]=CCD_con(x1,x2,alpha)

%% Parameters 
% N=length(x);
% n=floor(N/2);
n=length(x2);
alp=floor(alpha*(n+1))/(n+1); %Rounded Significant level


%% Bandwidth Selection and Model Training 
% idx=randsample(N,N);
% x1=x(idx(1:n));
% x2=x(idx(n+1:end));


H = logspace(-4,1,10);
opt_vol=Inf;
for h=H
    [~,~,vol]=conf_approx(x1,h,alpha);
    if vol<opt_vol
        opt_h=h;
        opt_vol=vol;
    end
end 
[g,cut,~]=conf_approx(x2,opt_h,alpha);

end 