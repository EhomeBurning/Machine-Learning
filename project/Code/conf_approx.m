function [g,cut,vol]=conf_approx(x,h,alpha) 
%Approximate conformal
%last update: 2017-12-05
  % y: 2 by n data matrix
  % h: bandwidth
  % grid: 1 by n_grid coordinate grids
  % alpha: siginficant level
  
  n=length(x);
  K_0=1/(sqrt(2*pi)*h);
  g=fitdist(x,'Kernel','BandWidth',h);
  p_x=pdf(g,x);
  p_sort=sort(p_x);
  cut=p_sort(floor((n+1)*alpha))-K_0/(n*h);
  [f,xi]=ksdensity(x,'BandWidth',h,'NumPoints',n);
  vol=0;
  for i=1:n-1
      if (f(i)>=cut) && (f(i+1)>=cut)
          vol=vol+xi(i+1)-xi(i);
      end 
  end 
end 