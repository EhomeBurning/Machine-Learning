function [g1,g2,g3,cut,alp]=CCD_con(x1,x2,alpha,grid_x,grid_y,grid_z)

%% Parameters 
% N=length(x);
% n=floor(N/2);
n=length(x2);
alp=floor(alpha*(n+1))/(n+1); %Rounded Significant level


%% Bandwidth Selection and Model Training 
% idx=randsample(N,N);
% x1=x(idx(1:n));
% x2=x(idx(n+1:end));


H = logspace(-1,0,10);
opt_vol=Inf;
for h=H
    [~,~,~,~,vol]=conf_approx_3D(x1,h,alpha,grid_x,grid_y,grid_z);
    if vol<opt_vol
        opt_h=h;
        opt_vol=vol;
    end
end 
[g1,g2,g3,cut,~]=conf_approx_3D(x2,opt_h,alpha,grid_x,grid_y,grid_z);

end 