import numpy as np;
import torch;
import matplotlib.pyplot as plt;
from typing import Tuple;

from Network import Neural_Network;
from Loss_Functions import PDE_Residual;



def Evaluate_Approx_Sol(
        Sol_NN  : Neural_Network,
        Coords  : torch.Tensor) -> np.array:
    """ This function evaluates the approximate solution at each row of
    Point_Coords. The only function that should call this is Setup_Axes. """

    # To limit memory usage, we evaluate the network in batches. The ith row of
    # Coords should hold the ith coordinate where we want to evaluate the
    # network. Thus, the number of Coordinates is the number of rows in Coords.
    Batch_Size : int = 5000;
    Num_Coords : int = Coords.shape[0];

    # Initialize an array to hold the value of the network at each coordinate.
    Approx_Sol = np.empty(Num_Coords, dtype = np.float32);

    # main loop.
    for i in range(0, Num_Coords - Batch_Size, Batch_Size):
        # Evaluate Sol_NN at this batch of coordinates.
        Batch_u = Sol_NN(Coords[i:(i + Batch_Size)]).squeeze();

        # Store these results in the appropriate components of nApprox_Sol.
        Approx_Sol[i:(i + Batch_Size)] = Batch_u.detach().numpy();

    # Clean up loop.
    Batch_u = Sol_NN(Coords[(i + Batch_Size):]).squeeze();
    Approx_Sol[(i + Batch_Size):] = Batch_u.detach().numpy();

    # All done! return!
    return Approx_Sol;



def Evaluate_Residual(
        Sol_NN          : Neural_Network,
        PDE_NN          : Neural_Network,
        Coords          : torch.Tensor,
        Data_Type       : torch.dtype,
        Device          : torch.device):
    """ This function evaluates the PDE residual at each row of Coords. The only
    function that should call this is Setup_Axes. """

    # To limit memory usage, we evaluate the Residual in batches. The ith row of
    # Coords should hold the ith coordinate where we want to evaluate the
    # network. Thus, the number of Coordinates is the number of rows of Coords.
    Batch_Size : int = 2000;
    Num_Coords : int = Coords.shape[0];

    # Initialize an array to hold the PDE residual at each coordinate.
    Residual = np.empty(Num_Coords, dtype = np.float32);

    # Main loop
    for i in range(0, Num_Coords - Batch_Size, Batch_Size):
        # Evaluate the PDE Residual for this batch of coordinates.
        Batch_Residual = PDE_Residual(
                            Sol_NN    = Sol_NN,
                            PDE_NN    = PDE_NN,
                            Coords    = Coords[i:(i+Batch_Size)],
                            Data_Type = Data_Type,
                            Device    = Device);

        # Store these in the appropriate components of Residual.
        Residual[i:(i + Batch_Size)] = Batch_Residual.detach().numpy();

    # Clean up loop.
    Batch_Residual = PDE_Residual(
                        Sol_NN    = Sol_NN,
                        PDE_NN    = PDE_NN,
                        Coords    = Coords[(i+Batch_Size):],
                        Data_Type = Data_Type,
                        Device    = Device)
    Residual[(i + Batch_Size):] = Batch_Residual.detach().numpy();

    # All done!
    return Residual;



def Initialize_Axes() -> Tuple[plt.figure, np.array]:
    """ This function sets up the figure, axes objects for plotting. There are
    many settings to tweak, so I thought the code would be cleaner if I hid
    those details in this function.

    ----------------------------------------------------------------------------
    Arguments:

    None!

    ----------------------------------------------------------------------------
    Returns:

    A two-element tuple. The first element contains the figure object. The
    second contains a NumPy array of axes objects (which should be passed to
    Update_Axes). """

    # Set up the figure object.
    fig = plt.figure(figsize = (9, 7));

    # Approximate solution subplot.
    Axes1 = fig.add_subplot(2, 2, 1);
    Axes1.set_title("Neural Network Approximation");

    # True solution subplot.
    Axes2 = fig.add_subplot(2, 2, 2);
    Axes2.set_title("True Solution");

    # Difference between the true and approximate solutions.
    Axes3 = fig.add_subplot(2, 2, 3);
    Axes3.set_title("Absolute Error");

    # Residual subplot.
    Axes4 = fig.add_subplot(2, 2, 4);
    Axes4.set_title("PDE Residual");

    # Package the axes objects into an array.
    Axes = np.array([Axes1, Axes2, Axes3, Axes4]);

    # Set settings that are the same for each Axes object.
    for i in range(4):
        # Set x, y bounds
        Axes[i].set_xbound(0., 1.);
        Axes[i].set_ybound(0., 1.);

        # This forces Python to produce a square plot.
        Axes[i].set_aspect('auto', adjustable = 'datalim');
        Axes[i].set_box_aspect(1.);

    return (fig, Axes);



def Setup_Axes(
        fig                 : plt.figure,
        Axes                : np.ndarray,
        Sol_NN              : Neural_Network,
        PDE_NN              : Neural_Network,
        x_points            : np.array,
        t_points            : np.array,
        True_Sol_On_Grid    : np.array,
        Torch_dtype         : torch.dtype = torch.float32,
        Device              : torch.device = torch.device('cpu')) -> None:
    """ This function makes four plots. One for the approximate solution, one
    for the true solution, one for their difference, and one for the PDE
    residual. x_points and t_points specify the domain of all four plots.

    Note: this function only works if u is a function of 1 spatial variable.

    ----------------------------------------------------------------------------
    Arguments:

    fig: The figure object to which the Axes belong. We need this to set up
    the color bars.

    Axes: The array of Axes object that we will plot on. Note that this
    function overwrites these axes.

    Sol_NN: The network that approximates the PDE solution.

    PDE_NN: The network that approximates the PDE.

    x_points, t_points: The set of possible x and t values, respectively. We
    use these to construct the grid of points that we plot on.

    True_Sol_at_Points: A NumPy array containing the true solution at each
    grid point (t, x coordinate). If t_points and x_points have n_t and n_x
    elements, respectively, then True_Sol_at_Points should be an n_x by n_t
    array whose i,j element holds the value of the true solution at t_points[j],
    x_points[i].

    Torch_dtype: All tensors in Sol_NN and PDE_NN should use this data type.

    Device: The device that Sol_NN and PDE_NN are loaded on.

    ----------------------------------------------------------------------------
    Returns:

    Nothing! """

    # First, construct the set of possible coordinates. grid_t_coords and
    # grid_x_coords are 2d NumPy arrays. Each row of these arrays corresponds to
    # a specific position. Each column corresponds to a specific time.
    grid_t_coords, grid_x_coords = np.meshgrid(t_points, x_points);

    # Flatten t_coords, x_coords. use them to generate grid point coodinates.
    flattened_grid_x_coords = grid_x_coords.reshape(-1, 1);
    flattened_grid_t_coords = grid_t_coords.reshape(-1, 1);
    Grid_Point_Coords = torch.from_numpy(np.hstack((flattened_grid_t_coords, flattened_grid_x_coords))).to(dtype = Torch_dtype, device = Device);

    # Get number of x and t values, respectively.
    n_x = len(x_points);
    n_t = len(t_points);

    # Put networks into evaluation mode.
    Sol_NN.eval();
    PDE_NN.eval();

    # Evaluate the network's approximate solution, the absolute error, and the
    # PDE residual at each coordinate. We need to reshape these into n_x by n_t
    # grids because that's what matplotlib's contour function wants.
    Approx_Sol_on_grid = Evaluate_Approx_Sol(Sol_NN, Grid_Point_Coords).reshape(n_x, n_t);
    Error_On_Grid      = np.abs(Approx_Sol_on_grid - True_Sol_On_Grid);
    Residual_on_Grid   = Evaluate_Residual(
                            Sol_NN    = Sol_NN,
                            PDE_NN    = PDE_NN,
                            Coords    = Grid_Point_Coords,
                            Data_Type = Torch_dtype,
                            Device    = Device).reshape(n_x, n_t);

    # Plot the approximate solution + color bar.
    ColorMap0 = Axes[0].contourf(grid_t_coords, grid_x_coords, Approx_Sol_on_grid, levels = 50, cmap = plt.cm.jet);
    fig.colorbar(ColorMap0, ax = Axes[0], fraction=0.046, pad=0.04, orientation='vertical');

    # Plot the true solution + color bar
    ColorMap1 = Axes[1].contourf(grid_t_coords, grid_x_coords, True_Sol_On_Grid, levels = 50, cmap = plt.cm.jet);
    fig.colorbar(ColorMap1, ax = Axes[1], fraction=0.046, pad=0.04, orientation='vertical');

    # Plot the Error between the true and approximate solution + color bar.
    ColorMap2 = Axes[2].contourf(grid_t_coords, grid_x_coords, Error_On_Grid, levels = 50, cmap = plt.cm.jet);
    fig.colorbar(ColorMap2, ax = Axes[2], fraction=0.046, pad=0.04, orientation='vertical');

    # Plot the residual + color bar
    ColorMap3 = Axes[3].contourf(grid_t_coords, grid_x_coords, Residual_on_Grid, levels = 50, cmap = plt.cm.jet);
    fig.colorbar(ColorMap3, ax = Axes[3], fraction=0.046, pad=0.04, orientation='vertical');

    # Set tight layout (to prevent overlapping... I have no idea why this isn't
    # a default setting. Matplotlib, you are special kind of awful).
    fig.tight_layout();
