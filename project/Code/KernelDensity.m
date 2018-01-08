%% 2-D kernel density
% last modified on 2011-09-24

function [p_Y p_grid] = KernelDensity(Y, h, is_grid, grid)
  % Y: 2 by n data matrix
  % h: bandwidth
  % grid: 2 by k array, the (grid) points of output
  % p_Y: the estimated density at the data points
  % p_grid: estimated density at grid points
  
  n = size(Y, 2);
  n_grid = size(grid, 2);
  p_Y = zeros(1, n);
  S = h^2*eye(2);
  for i = 1:n
    p_Y(i) = mean(mvnpdf(Y', Y(:, i)', S));
  end
  if is_grid
    p_1 = zeros(n_grid, n);
    p_2 = p_1;
    for i  =1:n_grid
      p_1(i, :) = normpdf(Y(1, :), grid(1, i), h);
      p_2(i, :) = normpdf(Y(2, :), grid(2, i), h);
    end
    p_grid = p_1 * p_2' / n;
  end
end