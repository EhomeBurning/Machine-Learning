function [g1,g2,g3,cut,vol]=conf_approx(x,h,alpha,grid_x,grid_y,grid_z) 
%Approximate conformal
%last update: 2017-12-05
  % y: 2 by n data matrix
  % h: bandwidth
  % grid_~: grid on the corresponded axis 
  % alpha: siginficant level
  
  %Calculate Parameters
  n=length(x);
  K_0=1/(sqrt(2*pi)*h)^3;
  m=length(grid_z);
  
  %% Kernel Density Estimator for each coordinate 
  g1=fitdist(x(:,1),'Kernel','BandWidth',h);
  g2=fitdist(x(:,2),'Kernel','BandWidth',h);
  g3=fitdist(x(:,3),'Kernel','BandWidth',h);
  
  p_x=pdf(g1,x(:,1)).*pdf(g2,x(:,2)).*pdf(g3,x(:,3));
  p_sort=sort(p_x);
  
  cut=p_sort(floor((n+1)*alpha))-K_0/(n*h^3);
  [f1,~]=ksdensity(x(:,1),grid_x,'BandWidth',h,'NumPoints',n);
  [f2,~]=ksdensity(x(:,2),grid_y,'BandWidth',h,'NumPoints',n);
  [f3,~]=ksdensity(x(:,3),grid_z,'BandWidth',h,'NumPoints',n);
  f=f3(1)*f1'*f2;
  for i=2:m
      f(:,:,i)=f1'*f2*f3(i);
  end 
  vol=sum(sum(sum(f>=cut)));
end 