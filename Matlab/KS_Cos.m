% KS equation: D_t U = -1.25(D_x^2 U) - 0.3125(D_x^4 U) - 2.5(D_x U)(U)
% with U(x, 0) = -cos(pi*x/10);


% Set up the problem domain. I want this to run on
%       (t, x) in [0,5] x [-5, 5]
x_l     = -5;
x_h     = 5;
t_l     = 0;
t_h     = 10;
Nt      = 251;
Domain  = [x_l, x_h];
T_span  = linspace(t_l, t_h, Nt);


% Set up the spinop for Burger's equation. For this problem:
%       L(u)    = -1.25(D_x^2 U) - .3125(D_x^4 U) 
%       N(u)    = -2.5(U)(D_x U).
%       Init(x) = -cos(pi*x/10)
S           = spinop(Domain, T_span);
S.lin       = @(u) -1.25*diff(u,2) - 0.3125*diff(u,4);
S.nonlin    = @(u) -1.25*diff(u.^2);
S.init      = chebfun(@(x) -cos(pi*x/10), Domain, 'vectorize');


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
