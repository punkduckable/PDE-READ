% Allen-Cahn equation: D_t U = 0.003 * D_x^2 U - U^3 + U

% Set up the problem domain. I want this to run on
%       (t, x) in [0, 10] x [-1, 1]
x_l     = -1;
x_h     = 1;
t_l     = 0;
t_h     = 10;
Nt      = 201;
Domain  = [x_l, x_h];
Tspan   = linspace(t_l, t_h, Nt);


% Set up the spinop for the Allen-Cahn equation. For this problem:
%       L(U)    = 0.003(D_x^2 U) + U 
%       N(U)    = -U^3.
%       Init(x) = 
S           = spinop(Domain, Tspan);
S.lin       = @(u) (0.003)*diff(u, 2) + u;
S.nonlin    = @(u) -u.^3;
S.init      = chebfun(@(x) 0.2*sin(2*pi*x).^5 + 0.8*sin(5*pi*x), Domain, 'vectorize');

% Solve!
disp("Solving...");
Nx      = 256;
Dt      = .001;
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
save('../Data/Allen_Cahn.mat','t','x','usol');

% Plot!
figure(1);
hold on;
set(gca, 'FontSize', 12);

pcolor(t, x, usol); shading interp, colorbar, axis tight, colormap(jet);

xlabel('time (s)');
ylabel('position (m)');
title("Allen-Cahn equation dataset");