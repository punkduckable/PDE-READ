import numpy;
import torch;
import matplotlib.pyplot as plt;
from   typing import Tuple;

from Settings_Reader    import Settings_Reader, Settings_Container;
from Plot_Dataset       import Data_Container, Load_Dataset;

# Nonsense to add Code diectory to the python search path.
import os;
import sys;

# Get path to parent directory
parent_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)));

# Add the Code directory to the python path.
Code_path   = os.path.join(parent_dir, "Code");
sys.path.append(Code_path);

from Network        import Neural_Network;
from Loss_Functions import PDE_Residual;



def Evaluate_Approx_Sol(
        Sol_NN  : Neural_Network,
        Coords  : torch.Tensor) -> numpy.array:
    """ This function evaluates the approximate solution on the coordinate
    in Coords. Coords should be a B by 2 tensor, where B is the batch size.
    The ith row of Coords should house the (t, x) coordinates of the ith
    coordinate.

    Note: Plot_Solution is the only function that should call this one. """

    # To limit memory usage, we evaluate the network in batches. The ith row of
    # Coords should hold the ith coordinate where we want to evaluate the
    # network. Thus, the number of Coordinates is the number of rows in Coords.
    Batch_Size : int = 1024;
    Num_Coords : int = Coords.shape[0];

    # Initialize an array to hold the value of the network at each coordinate.
    Approx_Sol : numpy.array = numpy.empty(Num_Coords, dtype = numpy.float32);

    # main loop.
    for i in range(0, Num_Coords - Batch_Size, Batch_Size):
        # Evaluate Sol_NN at this batch of coordinates.
        Batch_u : torch.Tensor = Sol_NN(Coords[i:(i + Batch_Size), :]).view(-1);

        # Store these results in the appropriate components of Approx_Sol.
        Approx_Sol[i:(i + Batch_Size)] = Batch_u.detach().numpy();

    # Clean up loop.
    Batch_u : torch.Tensor        = Sol_NN(Coords[(i + Batch_Size):, :]).view(-1);
    Approx_Sol[(i + Batch_Size):] = Batch_u.detach().numpy();

    # All done! return!
    return Approx_Sol;



def Evaluate_Residual(
        Sol_NN                      : Neural_Network,
        PDE_NN                      : Neural_Network,
        Time_Derivative_Order       : int,
        Spatial_Derivative_Order    : int,
        Coords                      : torch.Tensor):
    """ This function evaluates the PDE residual at each coordinate in Coords.
    Coords should be a B by 2 tensor, where B is the batch size.
    The ith row of Coords should house the (t, x) coordinates of the ith
    coordinate.

    Note: Plot_Solution is the only function that should call this one. """

    # To limit memory usage, we evaluate the Residual in batches. The ith row of
    # Coords should hold the ith coordinate where we want to evaluate the
    # network. Thus, the number of Coordinates is the number of rows of Coords.
    Batch_Size : int = 1024;
    Num_Coords : int = Coords.shape[0];

    # Initialize an array to hold the PDE residual at each coordinate.
    Residual : numpy.array = numpy.empty(Num_Coords, dtype = numpy.float32);

    # Main loop
    for i in range(0, Num_Coords - Batch_Size, Batch_Size):
        # Evaluate the PDE Residual for this batch of coordinates.
        Batch_Residual : torch.Tensor = PDE_Residual(
                            Sol_NN                      = Sol_NN,
                            PDE_NN                      = PDE_NN,
                            Time_Derivative_Order       = Time_Derivative_Order,
                            Spatial_Derivative_Order    = Spatial_Derivative_Order,
                            Coords                      = Coords[i:(i+Batch_Size), :]).view(-1);

        # Store these in the appropriate components of Residual.
        Residual[i:(i + Batch_Size)] = Batch_Residual.detach().numpy();

    # Clean up loop.
    Batch_Residual : torch.Tensor = PDE_Residual(
                        Sol_NN                      = Sol_NN,
                        PDE_NN                      = PDE_NN,
                        Time_Derivative_Order       = Time_Derivative_Order,
                        Spatial_Derivative_Order    = Spatial_Derivative_Order,
                        Coords                      = Coords[(i+Batch_Size):, :]).view(-1);
    Residual[(i + Batch_Size):] = Batch_Residual.detach().numpy();

    # All done! Return!
    return Residual;



def Initialize_Axes() -> Tuple[plt.figure, numpy.array]:
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
    Axes1.set_title("Training data set with noise");
    Axes1.set_xlabel("time (s)");
    Axes1.set_ylabel("position (m)");

    # True solution subplot.
    Axes2 = fig.add_subplot(2, 2, 2);
    Axes2.set_title("Neural Network Approximation");
    Axes2.set_xlabel("time (s)");
    Axes2.set_ylabel("position (m)");

    # Difference between the true and approximate solutions.
    Axes3 = fig.add_subplot(2, 2, 3);
    Axes3.set_title("Error with noise-free dataset");
    Axes3.set_xlabel("time (s)");
    Axes3.set_ylabel("position (m)");

    # Residual subplot.
    Axes4 = fig.add_subplot(2, 2, 4);
    Axes4.set_title("PDE Residual");
    Axes4.set_xlabel("time (s)");
    Axes4.set_ylabel("position (m)");

    # Package the axes objects into an array.
    Axes = numpy.array([Axes1, Axes2, Axes3, Axes4]);

    # Set settings that are the same for each Axes object.
    for i in range(4):
        # Set x, y bounds
        Axes[i].set_xbound(0., 1.);
        Axes[i].set_ybound(0., 1.);

        # This forces Python to produce a square plot.
        Axes[i].set_aspect('auto', adjustable = 'datalim');
        Axes[i].set_box_aspect(1.);

    return (fig, Axes);



def Plot_Solution(
        fig                         : plt.figure,
        Axes                        : numpy.ndarray,
        Sol_NN                      : Neural_Network,
        PDE_NN                      : Neural_Network,
        Time_Derivative_Order       : int,
        Spatial_Derivative_Order    : int,
        Data                        : Data_Container) -> None:
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

    Time_Derivative_Order: The order of the time derivative on the left-hand
    side of the PDE.

    Spatial_Derivative_Order: The highest order spatial derivatives of Sol_NN we
    need to evaluate.

    Data: This is a Data_Container object. It should contain four members (all
    of which are numpy arrays): x_points, t_points, Data_Set, and Noisy_Data_Set.
    x_points, t_points contain the set of possible x and t values, respectively.
    Data_Set and Noisy_Data_Set should contain the true solution with and
    without noise at each grid point (t, x coordinate). If t_points and x_points
    have n_t and n_x elements, respectively, then Data_Set and Noisy_Data_Set
    should be an n_x by n_t array whose i,j element holds the value of the true
    solution at t_points[j], x_points[i].

    ----------------------------------------------------------------------------
    Returns:

    Nothing! """

    # First, construct the set of possible coordinates. grid_t_coords and
    # grid_x_coords are 2d NumPy arrays. Each row of these arrays corresponds to
    # a specific position. Each column corresponds to a specific time.
    grid_t_coords, grid_x_coords = numpy.meshgrid(Data.t_points, Data.x_points);

    # Flatten t_coords, x_coords. use them to generate grid point coodinates.
    flattened_grid_x_coords = grid_x_coords.reshape(-1, 1);
    flattened_grid_t_coords = grid_t_coords.reshape(-1, 1);
    Grid_Point_Coords = torch.from_numpy(numpy.hstack((flattened_grid_t_coords, flattened_grid_x_coords)));

    # Get number of x and t values, respectively.
    n_x = len(Data.x_points);
    n_t = len(Data.t_points);

    # Put networks into evaluation mode.
    Sol_NN.eval();
    PDE_NN.eval();

    # Evaluate the network's approximate solution, the absolute error, and the
    # PDE residual at each coordinate. We need to reshape these into n_x by n_t
    # grids because that's what matplotlib's contour function wants.
    Approx_Sol_on_grid = Evaluate_Approx_Sol(Sol_NN, Grid_Point_Coords).reshape(n_x, n_t);
    Error_On_Grid      = numpy.abs(Approx_Sol_on_grid - Data.Data_Set);
    Residual_on_Grid   = Evaluate_Residual(
                            Sol_NN                      = Sol_NN,
                            PDE_NN                      = PDE_NN,
                            Time_Derivative_Order       = Time_Derivative_Order,
                            Spatial_Derivative_Order    = Spatial_Derivative_Order,
                            Coords                      = Grid_Point_Coords).reshape(n_x, n_t);

    # Plot the true solution + color bar.
    data_min : float = numpy.min(Data.Noisy_Data_Set);
    data_max : float = numpy.max(Data.Noisy_Data_Set);

    ColorMap0 = Axes[0].contourf(   grid_t_coords,
                                    grid_x_coords,
                                    Data.Noisy_Data_Set,
                                    levels = numpy.linspace(data_min, data_max, 500),
                                    cmap = plt.cm.jet);
    fig.colorbar(ColorMap0, ax = Axes[0], fraction=0.046, pad=0.04, orientation='vertical');

    # Plot the learned solution + color bar
    sol_min : float = numpy.min(Approx_Sol_on_grid);
    sol_max : float = numpy.max(Approx_Sol_on_grid);

    ColorMap1 = Axes[1].contourf(   grid_t_coords,
                                    grid_x_coords,
                                    Approx_Sol_on_grid,
                                    levels = numpy.linspace(sol_min, sol_max, 500),
                                    cmap = plt.cm.jet);
    fig.colorbar(ColorMap1, ax = Axes[1], fraction=0.046, pad=0.04, orientation='vertical');

    # Plot the Error between the approx solution and noise-free data set. + color bar.
    error_min : float = numpy.min(Error_On_Grid);
    error_max : float = numpy.max(Error_On_Grid);

    ColorMap2 = Axes[2].contourf(   grid_t_coords,
                                    grid_x_coords,
                                    Error_On_Grid,
                                    levels = numpy.linspace(error_min, error_max, 500),
                                    cmap = plt.cm.jet);
    fig.colorbar(ColorMap2, ax = Axes[2], fraction=0.046, pad=0.04, orientation='vertical');

    # Plot the residual + color bar
    resid_min : float = numpy.min(Residual_on_Grid);
    resid_max : float = numpy.max(Residual_on_Grid);

    ColorMap3 = Axes[3].contourf(   grid_t_coords,
                                    grid_x_coords,
                                    Residual_on_Grid,
                                    levels = numpy.linspace(resid_min, resid_max, 500),
                                    cmap = plt.cm.jet);
    fig.colorbar(ColorMap3, ax = Axes[3], fraction=0.046, pad=0.04, orientation='vertical');

    # Set tight layout (to prevent overlapping... I have no idea why this isn't
    # a default setting. Matplotlib, you are special kind of awful).
    fig.tight_layout();



if __name__ == "__main__":
    # First, read the settings
    (Settings, _) = Settings_Reader();

    # Next, load the dataset.
    Data = Load_Dataset(
                Data_Set_File_Name  = Settings.Data_Set_File_Name,
                Noise_Level         = Settings.Noise_Level);

    # Now, setup the networks.
    Sol_NN = Neural_Network( Num_Hidden_Layers   = Settings.Sol_Num_Hidden_Layers,
                             Neurons_Per_Layer   = Settings.Sol_Neurons_Per_Layer,
                             Input_Dim           = 2,
                             Output_Dim          = 1,
                             Activation_Function = Settings.Sol_Activation_Function);

    PDE_NN = Neural_Network( Num_Hidden_Layers   = Settings.PDE_Num_Hidden_Layers,
                             Neurons_Per_Layer   = Settings.PDE_Neurons_Per_Layer,
                             Input_Dim           = Settings.PDE_Spatial_Derivative_Order + 1,
                             Output_Dim          = 1,
                             Activation_Function = Settings.PDE_Activation_Function);

    Load_File_Path : str = "../Saves/" + Settings.Load_File_Name;
    Saved_State = torch.load(Load_File_Path, map_location=torch.device('cpu'));
    Sol_NN.load_state_dict(Saved_State["Sol_Network_State"]);
    PDE_NN.load_state_dict(Saved_State["PDE_Network_State"]);

    # Finally, make the plot.
    fig, Axes = Initialize_Axes();
    Plot_Solution(      fig                         = fig,
                        Axes                        = Axes,
                        Sol_NN                      = Sol_NN,
                        PDE_NN                      = PDE_NN,
                        Time_Derivative_Order       = Settings.PDE_Time_Derivative_Order,
                        Spatial_Derivative_Order    = Settings.PDE_Spatial_Derivative_Order,
                        Data                        = Data);

    # Show the plot and save it!
    plt.show();
    fig.savefig(fname = "../Figures/%s" % Settings.Load_File_Name);
