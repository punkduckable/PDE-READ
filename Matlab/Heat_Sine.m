% Set up the problem domain. I want this to work on the domain
% (x, t) in [0, 10] x [0, 10]
x_min   = 0;
x_max   = 10;
domain = [x_min x_max];

t_min = 0;
t_max = 10;
n_t   = 201;
tspan = linspace(t_min, t_max, n_t);

% Set up Heat equation operator
L = chebop(domain);
L.op = @(u) .05*diff(u, 2);
L.bc = 'periodic';

% Set initial condition
x_mid = (x_min + x_max)/2;
u0 = chebfun(@(x) sin(5*(x - x_min)*(2*pi/(x_max - x_min))), domain);

% Solve!
disp("Solving...");
u = expm(L, t_range, u0);

% Write usol to array
disp("Writing sol to array...");
usol = zeros(n_x - 1, n_t);
n_x     = 201;
x_range = linspace(x_min, x_max, n_x);
x_range = x_range(1:end - 1);

for i_t = 1:n_t
   usol(:, i_t) = u{i_t}(x_range);
end

% Save!
disp("Saving...");
t = t_range;
x = x_range;
save('../Data/Heat_Sine.mat','t','x','usol') 

% Plot.
disp("Plotting...");
LW = 'linewidth';
figure, waterfall(u, t_range, LW, 2)
view(10, 70), axis([x_min x_max t_min t_max -1 1])
