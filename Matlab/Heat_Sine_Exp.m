% Set up the spatial domain.
x_min   = 0;
x_max   = 10;
n_x     = 201;
x_range = linspace(x_min, x_max, n_x);
x_range = x_range(1:end - 1);
domain = [x_min x_max];

% Set up time domain.
t_min = 0;
t_max = 10;
n_t   = 201;
t_range = linspace(t_min, t_max, n_t);

% Set up Heat equation operator
L = chebop(domain);
L.op = @(u) .05*diff(u, 2);

% Set periorid BCs
L.bc = 'periodic';

% Set initial condition
x_mid = (x_min + x_max)/2;
u0 = chebfun(@(x) exp(-.5*(x - x_mid)^2)*sin(5*(x - x_min)*(2*pi/(x_max - x_min))), domain);

% Solve!
disp("Solving...");
u = expm(L, t_range, u0);

% Saving
disp("Saving...");
usol = zeros(n_x - 1, n_t);
for i_t = 1:n_t
   usol(:, i_t) = u{i_t}(x_range);
end
t = t_range;
x = x_range;
save('../Data/Heat_Sine_Exp.mat','t','x','usol') 

% Plotting.
disp("Plotting...");
LW = 'linewidth';
figure, waterfall(u, t_range, LW, 2)
view(10, 70), axis([x_min x_max t_min t_max -1 1])
