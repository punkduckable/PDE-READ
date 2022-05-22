% KdV equation: D_t U = -(D_x^3 U) - (D_x U)(U)
% with U(x, 0) = -sin(pi*x/20);


% Set up the problem domain. I want this to run on
%       (t, x) in [0,40] x [-10, 10]
x_l     = -20;
x_h     = 20;
t_l     = 0;
t_h     = 40;
Nt      = 201;
Domain  = [x_l, x_h];
T_span  = linspace(t_l, t_h, Nt);


% Set up the spinop for Burger's equation. For this problem:
%       L(u)    = -(D_x^3 U)
%       N(u)    = -(U)(D_x U).
%       Init(x) = -sin(pi*x/10)
S           = spinop(Domain, T_span);
S.lin       = @(u) -1.0*diff(u,3);
S.nonlin    = @(u) -0.5*diff(u.^2);
S.init      = chebfun(@(x) -sin(pi*x/20), Domain, 'vectorize');


% Solve!
disp("Solving...");
Nx      = 256;
Dt      = .0005;
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
t = T_span;
x = x_range;
save('../Data/KdV_Sine.mat','t','x','usol');

% Plot!
figure(1);
hold on;
set(gca, 'FontSize', 12);

pcolor(t, x, usol); shading interp, colorbar, axis tight, colormap(jet);

xlabel('time (s)');
ylabel('position (m)');
title("Korteweg–De Vries equation dataset");