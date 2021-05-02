import matplotlib.pyplot as plt;
import torch;
import numpy as np;
from typing import Tuple;

from Network import Neural_Network, PDE_Residual;



# Determine how well the network satisifies the PDE at a set of point.
def Evaluate_Residuals(u_NN : Neural_Network, N_NN : Neural_Network, Point_Coords : torch.Tensor) -> np.array:
    """ For brevity, let u = u_NN and N = N_NN. At each coordinate, this
    function computes
                    du/dt - N(u, du_dx,... )
    which we call the residual. If the NN perfectly satisified the PDE, then the
    residual would be zero everywhere. However, since the NN only approximates
    the PDE solution, we get non-zero residuals.

    ----------------------------------------------------------------------------
    Arguments:
    u_NN : the Neural Network that approximates solution to the learned PDE.

    N_NN : The Neural Network that approximates the PDE.

    Point_Coords : a tensor of coordinates of points where we want to evaluate
    the residual This must be a N by 2 tensor, where N is the number of points.
    The ith row of this tensor should contain the x,t coordinates of the ith
    point where we want to evaluate the residual.

    ----------------------------------------------------------------------------
    Returns:
    a numpy array. The ith element of this array gives the residual at the ith
    element of points. """

    # First, determine the number of points and intiailize the residual array.
    num_points : int = Point_Coords.shape[0];
    Residual = np.empty((num_points), dtype = np.float);

    for i in range(num_points):
        # Get the xy coordinate of the ith collocation point.
        xt = Point_Coords[i];

        # Evaluate the Residual from the learned PDE at this point.
        Residual[i] = PDE_Residual(u_NN, N_NN, xt).item();

    return Residual;



# Evaluate solution at a set of points.
def Evaluate_Approx_Sol(u_NN : Neural_Network, Point_Coords : torch.Tensor) -> np.array:
    """ This function evaluates the approximate solution at each element of
    Point_Coords.

    ----------------------------------------------------------------------------
    Arguments:
    u_NN : the Neural Network that approximates solution to the learned PDE.

    Point_Coords : The set of points where we want to evaluate the approximate
    solution. This should be a Nx2 tensor of floats whose ith row holds the x,t
    coordinates of the ith point where we want to evaluate the approximate
    solution.

    ----------------------------------------------------------------------------
    Returns:
    A numpy array whose ith element is the value of u_NN at the ith element of
    Point_Coords. If Point_Coords is a Nx2 tensor, then this is a N element
    numpy array. """

    # Get number of points, initialize the u array.
    num_Points : int = Point_Coords.shape[0];
    u_NN_at_Points = np.empty((num_Points), dtype = np.float);

    # Loop through the points, evaluate the network at each one.
    for i in range(num_Points):
        u_NN_at_Points[i] = u_NN.forward(Point_Coords[i]).item();

    return u_NN_at_Points;



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
    fig = plt.figure(figsize = (9, 7));

    # Approx solution subplot.
    Axes1 = fig.add_subplot(2, 2, 1);
    Axes1.set_title("Neural Network Approximation");

    # True solution subplot
    Axes2 = fig.add_subplot(2, 2, 2);
    Axes2.set_title("True Solution");

    # Difference between True and Approx solution
    Axes3 = fig.add_subplot(2, 2, 3);
    Axes3.set_title("Absolute Error");

    # Residual subplot.
    Axes4 = fig.add_subplot(2, 2, 4);
    Axes4.set_title("PDE Residual");

    # Package axes objects into an array.
    Axes = np.array([Axes1, Axes2, Axes3, Axes4]);

    # Set settings that are the same for each Axes object.
    # I set these parameters in a loop so that I only have to type them once,
    # thereby improving code maintainability.
    for i in range(4):
        # Set x, y bounds
        Axes[i].set_xbound(0., 1.);
        Axes[i].set_ybound(0., 1.);

        # Force python to produce a square plot.
        Axes[i].set_aspect('auto', adjustable = 'datalim');
        Axes[i].set_box_aspect(1.);

    return (fig, Axes);



# The plotting function!
def Update_Axes(fig                 : plt.figure,
                Axes                : np.ndarray,
                u_NN                : Neural_Network,
                N_NN                : Neural_Network,
                x_points            : np.array,
                t_points            : np.array,
                True_Sol_On_Grid    : np.array) -> None:
    """ This function plots the approximate solution and residual at the
    specified points.

    ----------------------------------------------------------------------------
    Arguments:
    fig : The figure object to which the Axes belong. We need this to set up
    the color bars.

    Axes : The array of Axes object that we will plot on. Note that this
    function will overwrite these axes.

    u_NN : the Neural Network that approximates solution to the learned PDE.

    N_NN : The Neural Network that approximates the PDE.

    x_points, t_points : The set of possible x and t values, respectively. We
    use this to construct the grid of points.

    True_Sol_at_Points : A numpy array containing the true solution at each
    possible x, t coordinate.

    ----------------------------------------------------------------------------
    Returns:
    Nothing! """

    # First, construct the set of possible coordinates.
    # grid_t_coords and grid_x_coords are 2d numpy arrays. Each row of these
    # arrays corresponds to a specific position. Each column corresponds to a
    # specific time.
    grid_t_coords, grid_x_coords = np.meshgrid(t_points, x_points);
    # You may wonder why we do this again, when we did it in the data loader.
    # The answer is memory. There are a lot of coordinates, and storing them
    # in memory is wasteful. We really only need these coordinates when loading
    # the data, and when plotting. Thus, we recreate the grid points here. Sure,
    # this means we do the same computations twice, but we only run them twice,
    # so they won't tank overall performance.

    # Flatten t_coods, x_coords. use them to generate grid point coodinates.
    flattened_grid_x_coords  = grid_x_coords.flatten()[:, np.newaxis];
    flattened_grid_t_coords  = grid_t_coords.flatten()[:, np.newaxis];
    Grid_Point_Coords = torch.from_numpy(np.hstack((flattened_grid_x_coords, flattened_grid_t_coords))).float();

    # Get number of possible x and t values, respectively.
    n_x = len(x_points);
    n_t = len(t_points);

    # Evaluate the network's approximate solution, the difference between the
    # true and approximate solutions, and the PDE residual at the
    # specified Points. We need to reshape these into n_x by n_t grids, because
    # that's what matplotlib's contour function wants.
    u_NN_on_Grid      = Evaluate_Approx_Sol(u_NN, Grid_Point_Coords).reshape(n_x, n_t);
    Error_On_Grid     = np.abs(u_NN_on_Grid - True_Sol_On_Grid);
    Residual_on_Grid  = Evaluate_Residuals(u_NN, N_NN, Grid_Point_Coords).reshape(n_x, n_t);

    # Plot the approximate solution + colorbar.
    ColorMap0 = Axes[0].contourf(grid_t_coords, grid_x_coords, u_NN_on_Grid, levels = 50, cmap = plt.cm.jet);
    fig.colorbar(ColorMap0, ax = Axes[0], fraction=0.046, pad=0.04, orientation='vertical');

    # Plot the true solution + colorbar
    ColorMap1 = Axes[1].contourf(grid_t_coords, grid_x_coords, True_Sol_On_Grid, levels = 50, cmap = plt.cm.jet);
    fig.colorbar(ColorMap1, ax = Axes[1], fraction=0.046, pad=0.04, orientation='vertical');

    # Plot the Error between the true and approximate solution + colorbar.
    ColorMap2 = Axes[2].contourf(grid_t_coords, grid_x_coords, Error_On_Grid, levels = 50, cmap = plt.cm.jet);
    fig.colorbar(ColorMap2, ax = Axes[2], fraction=0.046, pad=0.04, orientation='vertical');

    # Plot the residual + colorbar
    ColorMap3 = Axes[3].contourf(grid_t_coords, grid_x_coords, Residual_on_Grid, levels = 50, cmap = plt.cm.jet);
    fig.colorbar(ColorMap3, ax = Axes[3], fraction=0.046, pad=0.04, orientation='vertical');

    # Set tight layout (to prevent overlapping... I have no idea why this isn't
    # a default setting. Matplotlib, you are special kind of awful).
    fig.tight_layout();
