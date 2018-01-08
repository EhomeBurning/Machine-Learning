function [g1,g2,cut,vol]=conf_approx(x,h,alpha,grid_x,grid_y) 
%Approximate conformal
%last update: 2017-12-05
  % y: 2 by n data matrix
  % h: bandwidth
  % grid: 1 by n_grid coordinate grids
  % alpha: siginficant level
  
  n=length(x);
  K_0=1/(2*pi*h^2);
  g1=fitdist(x(:,1),'Kernel','BandWidth',h);
  g2=fitdist(x(:,2),'Kernel','BandWidth',h);
  p_x=pdf(g1,x(:,1)).*pdf(g2,x(:,2));
  p_sort=sort(p_x);
  cut=p_sort(floor((n+1)*alpha))-K_0/(n*h^2);
  [f1,~]=ksdensity(x(:,1),grid_x,'BandWidth',h,'NumPoints',n);
  [f2,~]=ksdensity(x(:,2),grid_y,'BandWidth',h,'NumPoints',n);
  f=f2'*f1;
  vol=sum(sum(f2'*f1>=cut));
end 