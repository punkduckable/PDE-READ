% Consider the function (t,x,y) -> exp(.05(t - x - y)) + sin(t-x) + sin(t-y)
% Some quick calculations reveal that this function satisifies the 
% 2D wave equation:
%           u_{tt} = u_{xx} + u_{yy}
% The purpose of this file is to effectively evaluate this function on a 
% grid of (t, x, y) points and save those points in a file, thereby
% creating a dataset for the wave equation with 2 spatial variables. 

% First, specify the problem domain. We will use (t, x, y) \in [0, 10] x
% [-5, 5] x [-5, 5].
disp("Setting up....");
t_l = 0;
t_h = 10;
x_l = -5;
x_h = 5;
y_l = -5;
y_h = 5;

xy_domain   = [x_l, x_h, y_l, y_h];
txy_domain  = [t_l, t_h, x_l, x_h, y_l, y_h];


% Define the function as a chebfun3 
u_cheb = chebfun3(@(t, x, y) sin(t - x) + exp(.05*(t - x - y)) + sin(t - y), txy_domain, 'vectorize');


% Now, make a grid of x, y, and t points at which to sample the function.
Nx = 64;
Ny = 64;
Nt = 201;

x_values = linspace(x_l, x_h, Nx);
y_values = linspace(y_l, y_h, Ny);
t_values = linspace(t_l, t_h, Nt);

x_coords = zeros(Nx, Ny);
y_coords = zeros(Nx, Ny);
for i = 1:Nx
    x = x_values(i);
    
    for j = 1:Ny
        y = y_values(j);
        
        x_coords(i, j) = x;
        y_coords(i, j) = y;
    end
end


% define a big array to hold the values of u at the various points. 
disp("Evaluating chebfun on (t, x, y) grid...");
usol = zeros(Nt, Nx, Ny);
for k = 1:Nt
    % Get the current t value
    t = t_values(k);
    
    % Set up a new chebfun2, which is u_cheb restricted to this time value.
    u_cheb_t = chebfun2(@(x, y) u_cheb(t, x, y), xy_domain, 'vectorize');
    
    % Evaluate u_cheb_t on the x/y_coords, store the result in usol.
    usol_t          = u_cheb_t(x_coords, y_coords);
    usol(k, :, :)   = usol_t;
end


% Save! 
disp("Saving...");
x       = x_values;
y       = y_values;
t       = t_values;
save('../Data/Wave_Sine_Exp_2D.mat','t','x', 'y','usol');