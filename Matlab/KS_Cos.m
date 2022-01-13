% KS equation: D_t U = 2.5(D_x^2 U) - .625(D_x^4 U) - 5(D_x U)(U)
% with U(x, 0) = -cos(pi*x/5);


% Set up the problem domain. I want this to run on
%       (t, x) in [0,5] x [-5, 5]
x_l     = -5;
x_h     = 5;
t_l     = 0;
t_h     = 5;
Nt      = 251;
Domain  = [x_l, x_h];
T_span  = linspace(t_l, t_h, Nt);


% Set up the spinop for Burger's equation. For this problem:
%       L(u)    = -(D_x^2 U) - (D_x^4 U) 
%       N(u)    = (U)(D_x U).
%       Init(x) = -sin(pi*x/10)
S           = spinop(Domain, T_span);
S.lin       = @(u) -2.5*diff(u,2) - 0.625*diff(u,4);
S.nonlin    = @(u) -2.5*diff(u.^2);
S.init      = chebfun(@(x) -cos(pi*x/5), Domain, 'vectorize');


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
save('../Data/KS_Cos.mat','t','x','usol');

% Plot!
figure(1);
hold on;
set(gca, 'FontSize', 12);

pcolor(t, x, usol); shading interp, colorbar, axis tight, colormap(jet);

xlabel('time (s)');
ylabel('position (m)');
title("Kuramoto–Sivashinsky equation dataset");
