%% main script for 2-d conformal
%  last updated by jinglei on 2011-09-24
%  The code below reproduces part of the 2-D kernel density conformal 
%  precition set simulation result in the paper of Lei, Robins and 
%  Wasserman (JASA 2013, Distribution Free Prediction Sets).

% parameters
n = 1000;
alpha = 0.1;
sig1 = 2;
sig2 = 0.5;
max1 = sqrt(2.2*log(n/2)) * sig1;
max2 = sqrt(2.2*log(n/2)) * sig2;
shift = max1 - 2;

grid = [(shift - max1):0.1:(shift+max1); (shift - max1):0.1:(shift+max1)];
cell_size = (grid(1,2) - grid(1,1))^2;

% generate truth
f1_1 = normpdf(grid(1,:), shift, sig1);
f1_2 = normpdf(grid(2,:), 0, sig2);
f1 = f1_1' * f1_2;

f2_1 = normpdf(grid(1,:), 0, sig2);
f2_2 = normpdf(grid(2,:), shift, sig1);
f2 = f2_1' * f2_2;

f = (f1 + f2 ) / 2;

f_max = max(max(f));
f_ind = 0:(f_max/1000):f_max;
i = 1;
g_new = 1;
new_set = f;
while g_new >= (1 - alpha)
    g = g_new;
    true_set = new_set;
    i = i + 1;
    new_set = (f >= f_ind(i));
    g_new = sum( sum( f(new_set) * 0.01 ) );
end

% data
m1 = binornd(n, 0.5);
Y1 = randn(2, m1);
Y2 = randn(2, n - m1);
Y1 = [sig1, 0; 0, sig2] * Y1;
Y1(1, :) = Y1(1, :) + shift;
Y2 = [sig2, 0; 0, sig1] * Y2;
Y2(2, :) = Y2(2, :) + shift;
Y = [Y1, Y2];

% density level set
bw_power = 0:0.5:4;
H = (log(n)/n)^(1/2) * 2.^(bw_power);
conf_set_size = zeros(3,length(H));
for i = 1:length(H)
    h = H(i);
    conf_set = conformal(Y, h, grid, alpha);
    [conf_inner, conf_outer] = conf_approx(Y, h, grid, alpha);
    conf_set_size(1,i) = sum(sum(conf_set)) * cell_size;
    conf_set_size(2,i) = sum(sum(conf_inner)) * cell_size;
    conf_set_size(3,i) = sum(sum(conf_outer)) * cell_size;
    %figure;contour(conf_set+conf_inner+conf_outer, [1, 2, 3])
    %hold on; contour(true_set, 1)
end
[ignore i_star] = min(conf_set_size, [], 2);
h_star = H(i_star);

conf_set = conformal(Y, 0.7, grid, alpha);
[conf_inner, conf_outer] = conf_approx(Y, 0.7, grid, alpha);

figure
% hold off;
contour(grid(1,:), grid(2, :), true_set, 1, 'LineColor', [0.5, 0.5,0.5], 'LineWidth', 2)
hold on;
contour(grid(1,:), grid(2, :), conf_outer, 1, 'LineColor', 'r', 'LineWidth', 2)
contour(grid(1,:), grid(2, :), conf_inner, 1, 'LineColor', 'g', 'LineWidth', 2)
contour(grid(1,:), grid(2, :), conf_set, 1, 'LineColor', 'b', 'LineWidth', 2)
plot(Y(1,:), Y(2,:), 'k.')
legend('Optimal Set', 'Outer Bound', 'Inner Bound', 'Conformal Set', 'Data Point')

figure;
plot(bw_power,conf_set_size(1,:),'LineWidth', 1.5, 'color', [0,0,1])
hold on
plot(bw_power,conf_set_size(2,:),'r','LineWidth', 1.5, 'color', [0,1,0])
plot(bw_power,conf_set_size(3,:),'r','Linewidth', 1.5, 'color', [1,0,0])
h = plot(bw_power(i_star(1))*[1,1],[0,max(conf_set_size(3,:))], 'Linewidth', 0.5, 'LineStyle', ':', 'color', [0.4, 0.4, 0.4]);
set(get(get(h, 'Annotation'), 'LegendInformation'), 'IconDisplayStyle', 'off');
