import matplotlib.pyplot as plt;
import torch;
import numpy as np;
from typing import Tuple;

from Network import Neural_Network;



# True solution to Poission's equation with the above driving term.
def True_Solution(xy : torch.Tensor) -> float:
    """ Evaluates the true (known) solution to the PDE at a particular point.

    ----------------------------------------------------------------------------
    Arguments:
    xy : a 2 element Tensor containing the coordinates we want to evaluate the
    true solution at. The first element is the x coordinate while the second
    is the y coordinate. Both elements should be in [0,1].

    ----------------------------------------------------------------------------
    Returns:
    A scalar (single element) tensor containing the true solution at (x, y). """

    # Extract coordinates
    x = xy[0].item();
    y = xy[1].item();

    # Return true solution at these coordinates!
    return (np.sin(np.pi * x) * np.sin(np.pi * y));



# Residual function to determine how well the network satisifies the PDE.
def PDE_Residual(u_NN : Neural_Network, Points : torch.Tensor) -> np.array:
    """ Let u_NN denote the approximate PDE solution created by the Neural
    Network. This function computes (d^2 u_NN)/dx^2 + (d^2 u_NN)/dy^2 + f, which
    we call the residual, at each element of points. If the NN perfectly
    satisified the PDE, then the residual would be zero everywhere. However,
    since the NN only approximates the PDE solution, we get non-zero residuals.

    ----------------------------------------------------------------------------
    Arguments:
    u_NN : the Neural Network that approximates the PDE solution.

    Points : a tensor of coordinates of points where we want to evaluate the
    residual This must be a N by 2 tensor, where N is the number of points. The
    ith row of this tensor should contain the x,y coordinates of the ith point
    where we want to evaluate the residual. Each element of Points should be in
    [0,1]x[0,1]

    ----------------------------------------------------------------------------
    Returns:
    a numpy array. The ith element of this array gives the residual at the ith
    element of points. """

    # First, determine the number of points and intiailize the residual array.
    num_points : int = Points.shape[0];
    Residual = np.empty((num_points), dtype = np.float);

    for i in range(num_points):
        # Get the xy coordinate of the ith point.
        xy = Points[i];

        # we need to evalute the PDE, which means computing derivatives of
        # the approximate solution with respect to x and y. Thus, xy requires
        # a gradient.
        xy.requires_grad_(True);

        # Compute the neural network approximation of the solution at this
        # point.
        u = u_NN.forward(xy);

        # Compute the gradient of u with respect to the input coordinates
        # x and y. This will yield a 2 element tensor, whose first element
        # is du/dx and whose second element is du/dy. We will need the
        # graph used to compute grad_u when evaluating the second derivatives,
        # so we set create_graph = True.
        grad_u = torch.autograd.grad(u, xy, retain_graph = True, create_graph = True)[0];

        # extract the partial derivatives
        du_dx = grad_u[0];
        du_dy = grad_u[1];

        # Compute the gradient of du_dx and du_dy with respect to the input
        # coordinates. The 0 component of grad_du_dx holds d^2u/dx^2 while
        # the 1 component of grad_du_dy holds d^2u/dy^2. We need to keep the
        # graph of grad_u to evaluate du_dy, so we set retain_graph to true
        # when computing grad_du_dx. We don't plan to do any backpropigation,
        # so we don't need to compute a new graph for the second gradients. We
        # don't need the graph after computing the derivatives, so we
        # set retain_graph = False in the second call (which will free the graphs
        # for grad_u and u).
        grad_du_dx = torch.autograd.grad(du_dx, xy, retain_graph = True)[0];
        grad_du_dy = torch.autograd.grad(du_dy, xy)[0];
        d2u_dx2 = grad_du_dx[0];
        d2u_dy2 = grad_du_dy[1];

        # Now, determine the residual and put it in the corresponding element of
        # the residual array.
        Residual[i] = (d2u_dx2 + d2u_dy2).item();

    return Residual;



# Evaluate solution at a set of points.
def Evaluate_NN(u_NN : Neural_Network, Points : torch.Tensor) -> np.array:
    """ This function evaluates the neural network at each element of Points.

    ----------------------------------------------------------------------------
    Arguments:
    u_NN : The neural network that (should) give an approximate solution to
    Poisson's equation.

    Points : The set of points we want to evaluate the solution at. This should
    be a Nx2 tensor of floats whose ith row holds the x,y coordinates of the
    ith point we want to evaluate the network at. Each element of Points
    should be an element of [0,1]x[0,1].

    ----------------------------------------------------------------------------
    Returns:
    A numpy array whose ith element is the value of u_NN at the ith element of
    Points. If Points is a Nx2 tensor, then this is a N element numpy array. """

    # Get number of points, initialize the u array.
    num_Points : int = Points.shape[0];
    u_NN_at_Points = np.empty((num_Points), dtype = np.float);

    # Loop through the points, evaluate the network at each one.
    for i in range(num_Points):
        u_NN_at_Points[i] = u_NN.forward(Points[i]).item();

    return u_NN_at_Points;



# Evaluate the true solution.
def Evaluate_True_Solution(Points : torch.Tensor) -> np.array:
    """ This function evaluates the actual solution at each point of Points.

    ----------------------------------------------------------------------------
    Arguments:
    Points : The set of points where we want to evaluate the solution. This
    should be an Nx2 tensor, whose ith row holds the x,y coordinates of the ith
    point we want to evaluate the true solution at. Each element of Points
    should be an element of [0,1]x[0,1].

    ----------------------------------------------------------------------------
    Returns:
    A numpy array whose ith element is the value of the True Solution at the ith
    element of Points. If Points is a Nx2 tensor, then this is a N element numpy
    array. """

    # Get number of points, initialize the u array.
    num_Points : int = Points.shape[0];
    True_Sol_at_Points = np.empty((num_Points), dtype = np.float);

    # Loop through the points, evaluate the network at each one.
    for i in range(num_Points):
        True_Sol_at_Points[i] = True_Solution(Points[i]).item();

    return True_Sol_at_Points;



# Generate plotting gridpoints.
def Generate_Plot_Gridpoints(n : int) -> torch.Tensor:
    """ This function generates a uniformly spaced grid of points in [0,1]x[0,1]
    for plotting purposes. Sure, you could do this with meshgrid, but I prefer
    to write my own code, especially if it won't impact performance by much
    (the code should only call this function once). I like explicit code.

    ----------------------------------------------------------------------------
    Arguments:

    n : the number of grid points in each dimension. This function will generate
    an n by n grid of points in the unit (2) square.

    ----------------------------------------------------------------------------
    Returns:

    an n^2 x 2 tensor of points. The ith row of this tensor holds the x,y
    coordinates of the ith point. The zero element has coordinates (0,0),
    the 1 element has coordinates (0, 1/(n - 1))... the n element has coordinates
    (1/(n - 1), 0)... the n*n - 1 element has coordinates (1, 1). """

    Points = torch.empty((n*n, 2), dtype = torch.float);

    # Using the rule defined above, we can see that the i*n + j element of
    # Points should have coordinates j/(n - 1), i/(n - 1). (think about it)
    for i in range(n):
        for j in range(n):
            Points[n*i + j, 0] = i/(n - 1);
            Points[n*i + j, 1] = j/(n - 1);

    return Points;



# Set up Axes objects for plotting
def Setup_Axes() -> Tuple[plt.figure, np.array]:
    """ This function sets up the figure, axes objects for plotting. There
    are a lot of settings to tweak, so I thought the code would be cleaner
    if those details were outsourced to this function.

    ----------------------------------------------------------------------------
    Arguments:
    None!

    ----------------------------------------------------------------------------
    Returns:
    A tuple. The first element contains the figure object, the second contains
    a numpy array of axes objects (to be passed to Update_Axes). """

    # Set up the figure object.
    fig = plt.figure(figsize = (12, 4));

    # Approx solution subplot.
    Axes1 = fig.add_subplot(1, 3, 1);
    Axes1.set_title("Neural Network Approximation");

    # True solution subplot
    Axes2 = fig.add_subplot(1, 3, 2);
    Axes2.set_title("True Solution");

    # Residual subplot.
    Axes3 = fig.add_subplot(1, 3, 3);
    Axes3.set_title("PDE Residual");

    # Package axes objects into an array.
    Axes = np.array([Axes1, Axes2, Axes3]);

    # Set settings that are the same for each Axes object.
    # The domain of each Axes object is the unit (2) square, and it's aspect
    # ratio should be equal. I set these parameters in a loop so that I only
    # have to type them once, thereby improving code maintainability.
    for i in range(3):
        # Set x, y bounds
        Axes[i].set_xbound(0., 1.);
        Axes[i].set_ybound(0., 1.);

        # Force python to produce a square plot.
        Axes[i].set_aspect('equal', adjustable = 'datalim');
        Axes[i].set_box_aspect(1.);

    return (fig, Axes);



# The plotting function!
def Update_Axes(fig : plt.figure, Axes : np.ndarray, u_NN : Neural_Network, Points : torch.Tensor, n : int) -> None:
    """ This function plots the approximate solution and residual at the
    specified points.

    ----------------------------------------------------------------------------
    Arguments:
    fig : The figure object to which the Axes belong. We need this to set up
    the color bars.

    Axes : The array of Axes object that we will plot on. Note that this
    function will overwrite these axes.

    u_NN : The neural network that gives the approximate solution to Poisson's
    equation.

    Points : The set of points we want to evaluate the approximate and true
    solutions, as well as the PDE Residual. This should be an (n*n)x2 tensor,
    whose ith row holds the x,y coordinates of the ith point we want to plot.
    Each element of Points should be an element of [0,1]x[0,1].

    n : the number of gridpoints along each axis. Points should be an n*n x 2
    tensor.

    ----------------------------------------------------------------------------
    Returns:
    Nothing! """

    # First, evaluate the network's approximate solution, the true solution, and
    # the PDE residual at the specified Points. We need to reshape these into
    # nxn grids, because that's what matplotlib's contour function wants. It's
    # annoying, but it is what it is.
    u_NN_at_Points      = Evaluate_NN(u_NN, Points).reshape(n,n);
    True_Sol_at_Points  = Evaluate_True_Solution(Points).reshape(n,n);
    Residual_at_Points  = PDE_Residual(u_NN, Points).reshape(n,n);

    # Extract the x and y coordinates of points, as np arrays. We also need to
    # reshape these as nxn grids (same reason as above.
    x = Points[:, 0].numpy().reshape(n,n);
    y = Points[:, 1].numpy().reshape(n,n);

    # Plot the approximate solution + colorbar.
    ColorMap0 = Axes[0].contourf(x, y, u_NN_at_Points.reshape(n,n), levels = 50, cmap = plt.cm.jet);
    fig.colorbar(ColorMap0, ax = Axes[0], fraction=0.046, pad=0.04, orientation='vertical');

    # Plot the true solution + colorbar
    ColorMap1 = Axes[1].contourf(x, y, True_Sol_at_Points.reshape(n,n), levels = 50, cmap = plt.cm.jet);
    fig.colorbar(ColorMap1, ax = Axes[1], fraction=0.046, pad=0.04, orientation='vertical');

    # Plot the residual + colorbar
    ColorMap2 = Axes[2].contourf(x, y, Residual_at_Points.reshape(n,n), levels = 50, cmap = plt.cm.jet);
    fig.colorbar(ColorMap2, ax = Axes[2], fraction=0.046, pad=0.04, orientation='vertical');

    # Set tight layout (to prevent overlapping... I have no idea why this isn't
    # a default setting. Matplotlib, you are special kind of awful).
    fig.tight_layout();
