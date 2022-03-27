% In 1D, the beam equation is: U_{tt} = -0.1*U_{xxxx}
% We solve this equation with fixed (no displacement/bending) BCs.

% Set up the problem domain. I want this to work on the domain
% (x, t) in [-5, 5] x [0, 20]
x_l         = -5.;
x_h         = 5.;
t_l         = 0.;
t_h         = 20.;
Domain      = [x_l, x_h, t_l, t_h];

% Set up the PDE operator.
%                          D_tt U        + 0.1*D_xxxx U
L           = chebop2(@(u) diff(u, 2, 1) + 0.1*diff(u, 4, 2), Domain);

% Set BCs. We clamp both ends (no displacement, or angle)
L.lbc       = @(x, u) [u;  diff(u, 1)];
L.rbc       = @(x, u) [u;  diff(u, 1)];

% Set IC.
L.dbc       = @(x, u) [u - 1.0*exp(-.5*(x - 2.5).^2); diff(u)];

% Solve the following PDE: U_tt = -0.1*U_xxxx subject to the above IC
disp("Solving....");
U_Cheb = L \ 0;


% Write usol to array
disp("Writing sol to array...");

Nx       = 201;
Nt       = 201;
X_Values = linspace(x_l, x_h, Nx);
T_Values = linspace(t_l, t_h, Nt);
usol = zeros(Nx, Nt);

for j = 1:Nt
    usol(:, j) = real(U_Cheb(X_Values, T_Values(j)));
end

% Save!
disp("Saving...");
t = T_Values;
x = X_Values;
save('../Data/Beam_Exp.mat','t','x','usol')

% Plot.
disp("Plotting...");
figure(1);
hold on;
set(gca, 'FontSize', 12);

pcolor(t, x, usol); shading interp, colorbar, axis tight, colormap(jet);

xlabel('time (s)');
ylabel('position (m)');
title("Beam equation dataset");
