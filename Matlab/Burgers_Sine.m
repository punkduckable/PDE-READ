% Burgers equation: D_t U = (D_x U)(U) + .1(D_x^2 U).


% Set up the problem domain. I want this to run on
%       (t, x) in [0,10] x [-8, 8]
x_l     = -8;
x_h     = 8;
t_l     = 0;
t_h     = 10;
Nt      = 201;
Domain  = [x_l, x_h];
Tspan   = linspace(t_l, t_h, Nt);


% Set up the spinop for Burger's equation. For this problem:
%       L(u)    = .1 (D_x^2 U) and 
%       N(u)    = (U)(D_x U).
%       Init(x) = -sin(pi*x/8)
S           = spinop(Domain, Tspan);
S.lin       = @(u) 0.1*diff(u, 2);
S.nonlin    = @(u) -0.5*diff(u.^2);
S.init      = chebfun(@(x) -sin(pi*x/8), Domain, 'vectorize');


% Solve!
disp("Solving...");
Nx      = 256;
Dt      = .0002;
U_Cheb  = spin(S, Nx, Dt, 'plot', 'off');


% Make the dataset....
disp("Writing solution to array...");
usol    = zeros(Nx, Nt);
x_vals  = linspace(x_l, x_h, Nx + 1);
x_range = x_vals(1:(end - 1));
for t = 1:Nt
    usol(:, t) = U_Cheb{t}(x_range);
end

% Save!
disp("Saving...");
t = Tspan;
x = x_range;
save('../Data/Burgers_Sine.mat','t','x','usol');

% Plot!
figure(1);
hold on;
set(gca, 'FontSize', 12);

pcolor(t, x, usol); shading interp, colorbar, axis tight, colormap(jet);

xlabel('time (s)');
ylabel('position (m)');
title("Burgers' equation dataset (Sine IC)");