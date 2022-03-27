% Klein-Gordon equation: U_tt = .5*U_xx - 5*U
% We solve this equation with 0 BCs

% First, set up the problem domain.
x_l = -1;
x_h = 1;
t_l = 0;
t_h = 3;
Domain = [x_l, x_h, t_l, t_h];

% Next, set up the operator
%                u_tt        - .5*u_xx        + 5*u = 0
L = chebop2(@(u) diff(u,2,1) - .5*diff(u,2,2) + 5*u, Domain); 
L.lbc = @(x, u) u; 
L.rbc = @(x, u) u; 
L.dbc = @(x,u) [u - exp(-20*(x).^2) ; diff(u)];
u = L \ 0; 

% Now solve!
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
save('../Data/KG_Exp.mat','t','x','usol') 

% Plot.
figure(1);
hold on;
set(gca, 'FontSize', 12);

pcolor(t, x, usol); shading interp, colorbar, axis tight, colormap(jet);

xlabel('time (s)');
ylabel('position (m)');
title("Klein-Gordon equation dataset");

