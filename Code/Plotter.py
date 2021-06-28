import numpy as np;
import torch;
import matplotlib.pyplot as plt;
from typing import Tuple;

from Network import Neural_Network;
from Loss_Functions import PDE_Residual;



def Evaluate_Approx_Sol(
        u_NN         : Neural_Network,
        Point_Coords : torch.Tensor) -> np.array:
    """ This function evaluates the approximate solution at each element of
    Point_Coords.

    Note: This function works regardless of how many spatial variables u depends
    on.

    ----------------------------------------------------------------------------
    Arguments:

    u_NN: The network that approximates the PDE solution.

    Point_Coords: The set of points where we want to evaluate the approximate
    solution. If u accepts d spatial vaiables, then this should be a d+1 column
    tensor whose ith column holds the t, x_1,... x_d coordinates of the ith
    point where we want to evaluate the approximate solution.

    ----------------------------------------------------------------------------
    Returns:

    A numpy array whose ith element holds the value of u_NN at the ith point.
    """

    # Evaluate the Netowk at each point, return its numpy equivalent.
    u_approx_at_points = u_NN(Point_Coords);
    return u_approx_at_points.detach().numpy();



def Initialize_Axes() -> Tuple[plt.figure, np.array]:
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



def Setup_Axes(
        fig                 : plt.figure,
        Axes                : np.ndarray,
        u_NN                : Neural_Network,
        N_NN                : Neural_Network,
        x_points            : np.array,
        t_points            : np.array,
        u_true_On_Grid      : np.array,
        Torch_dtype         : torch.dtype = torch.float32,
        Device              : torch.device = torch.device('cpu')) -> None:
    """ This function plots the approximate solution and residual at the
    specified points.

    Note: this function only works is u is a function of 1 spatial variable.

    ----------------------------------------------------------------------------
    Arguments:

    fig: The figure object to which the Axes belong. We need this to set up
    the color bars.

    Axes: The array of Axes object that we will plot on. Note that this
    function will overwrite these axes.

    u_NN: The network that approximates the PDE solution.

    N_NN: The Neural Network that approximates the PDE.

    x_points, t_points: The set of possible x and t values, respectively. We
    use this to construct the grid of points.

    True_Sol_at_Points: A numpy array containing the true solution at each
    possible x, t coordinate.

    Torch_dtype: The data type that all tensors use. All tensors in u_NN and N_NN
    should use this data type.

    Device: The device that u_NN and N_NN are loaded on.

    ----------------------------------------------------------------------------
    Returns:

    Nothing! """

    # First, construct the set of possible coordinates.
    # grid_t_coords and grid_x_coords are 2d numpy arrays. Each row of these
    # arrays corresponds to a specific position. Each column corresponds to a
    # specific time.
    grid_t_coords, grid_x_coords = np.meshgrid(t_points, x_points);

    # Flatten t_coods, x_coords. use them to generate grid point coodinates.
    flattened_grid_x_coords = grid_x_coords.flatten()[:, np.newaxis];
    flattened_grid_t_coords = grid_t_coords.flatten()[:, np.newaxis];
    Grid_Point_Coords = torch.from_numpy(np.hstack((flattened_grid_t_coords, flattened_grid_x_coords))).to(dtype = Torch_dtype, device = Device);

    # Get number of possible x and t values, respectively.
    n_x = len(x_points);
    n_t = len(t_points);

    # Evaluate the network's approximate solution, the absolute error, and the
    # PDE resitual at each coordinate. We need to reshape these into n_x by n_t
    # grids, because that's what matplotlib's contour function wants.
    u_approx_on_grid   = Evaluate_Approx_Sol(u_NN, Grid_Point_Coords).reshape(n_x, n_t);
    Error_On_Grid      = np.abs(u_approx_on_grid - u_true_On_Grid);
    Residual_on_Grid_1 = PDE_Residual(
                            u_NN      = u_NN,
                            N_NN      = N_NN,
                            Coords    = Grid_Point_Coords,
                            Data_Type = Torch_dtype,
                            Device    = Device);
    Residual_on_Grid   = Residual_on_Grid_1.detach().numpy().reshape(n_x, n_t);

    # Plot the approximate solution + colorbar.
    ColorMap0 = Axes[0].contourf(grid_t_coords, grid_x_coords, u_approx_on_grid, levels = 50, cmap = plt.cm.jet);
    fig.colorbar(ColorMap0, ax = Axes[0], fraction=0.046, pad=0.04, orientation='vertical');

    # Plot the true solution + colorbar
    ColorMap1 = Axes[1].contourf(grid_t_coords, grid_x_coords, u_true_On_Grid, levels = 50, cmap = plt.cm.jet);
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
