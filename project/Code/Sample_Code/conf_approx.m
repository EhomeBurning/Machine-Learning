function [conf_inner conf_outer] = conf_approx(Y, h, grid, alpha)
%Approximate conformal
%last update: 2011-09-22
  % Y: 2 by n data matrix
  % h: bandwidth
  % grid: 2 by n_grid coordinate grids
  % alpha: level
  
  n = size(Y, 2);
  K_0 = mvnpdf([0, 0], [0, 0], h^2 * eye(2));
  
  [p_Y, p_grid] = KernelDensity(Y, h, 1, grid);
  p_sort = sort(p_Y);
  cut_inner = p_sort(n * alpha);
  cut_outer = cut_inner - 1/(n*h^2)* K_0;
  conf_inner = (p_grid >= cut_inner);
  conf_outer = (p_grid >= cut_outer);
end

