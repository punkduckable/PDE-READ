% Allen-Cahn equation: D_t U = 0.05 D_x^2 U - 10*U^3 + 10*U

% Set up the problem domain. I want this to run on
%       (t, x) in [0, 50] x [0, 2*pi]
x_l     = 0;
x_h     = 2*pi;
t_l     = 0;
t_h     = 50;
Nt      = 201;
Domain  = [x_l, x_h];
Tspan   = linspace(t_l, t_h, Nt);


% Set up the spinop for the Allen-Cahn equation. For this problem:
%       L(U)    = 0.5(D_x^2 U) + 100*U 
%       N(U)    = -100*U^3.
%       Init(x) = 1/3*tanh(2*sin(x)) - exp(-23.5*(x-pi/2)^2) + 
%                 exp(-27*(x-4.2)^2) + exp(-38*(x-5.4)^2)
S           = spinop(Domain, Tspan);
S.lin       = @(u) (0.05)*diff(u, 2) + 10*u;
S.nonlin    = @(u) -10*u.^3;
S.init      = chebfun(@(x) (1/3)*tanh(2*sin(x)) - exp(-23.5*(x - pi/2).^2) + exp(-27*(x - 4.2).^2) + exp(-38*(x - 5.4).^2), Domain, 'vectorize');

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