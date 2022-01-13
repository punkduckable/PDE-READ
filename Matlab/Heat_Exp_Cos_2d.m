% 2D Heat equation: D_t U = .1(D_x^2 U + D_y^2 U).
% with U(0, x, y) = 10*exp(-(x^2 + y^2)/4)*cos(x*y).


% Set up the problem domain. I want this to work on the domain 
%       (t, x, y) in [0, 5] x [-5, 5] x [-5, 5]
x_l = -5;
x_h = 5;
y_l = -5;
y_h = 5;
domain      = [x_l, x_h, y_l, y_h];
Nt          = 101;
t_span       = linspace(0, 5, Nt);

% Now set up the spinop2 operator. For this problem:
%       L(u) = .1*Lap(U)
%       N(u) = 0
S           = spinop2(domain, t_span);
S.lin       = @(u) .1*lap(u);
S.nonlin    = @(u) 0*u;
S.init      = chebfun2(@(x, y) 10*exp(-(x^2 + y^2)/4)*cos(x*y), domain, 'vectorize');


% Solve the PDE! 
disp("Solving...");
dt          = .05;
Nxy         = 64;
U_cheb      = spin2(S, Nxy, dt, 'plot', 'off');


% Generate a grid of x, y coordinates. 
disp("Generting Coords...");
x_values    = linspace(x_l, x_h, Nxy);
y_values    = linspace(x_l, x_h, Nxy);

x_coords    = zeros(Nxy, Nxy);
y_coords    = zeros(Nxy, Nxy);
for i = 1:Nxy
    x = x_values(i);

    for j = 1:Nxy
        y = y_values(j);
        
        x_coords(i, j) = x;
        y_coords(i, j) = y;
    end 
end


% Store the solution in a big array. 
% For each t value, we evaluate U_cheb{t} on the grid. This returns an 
% array whose i,j entry is U_cheb{t}(x_values(i, j), y_values(i, j)). This 
% approach is a lotttt faster than if we used multiple loops. 
disp("Storing solution in array...");
usol        = zeros(Nt, Nxy, Nxy);
for t = 1:Nt
    usol(t, :, :) = U_cheb{t}(x_coords, y_coords);
end


% Save! 
disp("Saving...");
x       = x_values;
y       = y_values;
t       = t_span;
save('../Data/Heat_Exp_Cos_2D.mat','t','x', 'y','usol');