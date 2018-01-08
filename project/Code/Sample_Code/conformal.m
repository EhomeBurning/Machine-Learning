%% conformal prediction region
% last modified on 2011-09-23

function conf_set = conformal(Y, h, grid, alpha)
  % Y: 2 by n data matrix
  % h: bandwidth
  % grid: 2 by n_grid coordinate grids
  % alpha: level
  
  n = size(Y, 2);
  n_grid = size(grid, 2);
  conf_set = zeros(n_grid, n_grid);
  p_value = conf_set;
  
  [p_Y p_grid] = KernelDensity(Y, h, 1, grid);
  S = h^2 * eye(2);
  K_0 = mvnpdf([0, 0], [0, 0], S);
  for i = 1:n_grid
      for j = 1:n_grid
          y = [grid(1, i), grid(2, j)]';
          p_y = n/(n+1) * p_grid(i, j) + K_0/(n+1);
          p_y_Y = n/(n+1) * p_Y + 1/(n+1) * mvnpdf(Y', y', S)';
          p_value(i, j) = (sum(p_y_Y <= p_y) + 1) / (n+1);
      end
  end
  conf_set = (p_value >= alpha);
end