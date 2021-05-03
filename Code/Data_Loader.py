import numpy as np;
import torch;
import scipy.io;

from typing import Tuple;



class Data_Container:
    """ This class is a container for the data read in by the data loader.
    This data is used in various parts of the program.

    All "Coords" members should be two dimensional tensors/arrays with
    two columns (and an arbitrary number of rows). The 0 column should
    hold x-coordinates, while the 1 column should hold t-coordinates.

    All "Data" members should be one dimensional tensors/arrays. """

    def __init__(   self,
                    x_points            : np.array,
                    t_points            : np.array,
                    True_Sol_On_Grid    : np.array,
                    IC_Coords           : torch.Tensor,
                    IC_Data             : torch.Tensor,
                    Lower_Bound_Coords  : torch.Tensor,
                    Upper_Bound_Coords  : torch.Tensor,
                    Train_Data_Coords   : torch.Tensor,
                    Train_Data_Values   : torch.Tensor,
                    Train_Coloc_Coords  : torch.Tensor,
                    Test_Coloc_Coords   : torch.Tensor,
                    Test_Data_Values    : torch.Tensor,
                    Test_Data_Coords    : torch.Tensor ):

        # Plotting.
        self.x_points           = x_points;
        self.t_points           = t_points;
        self.True_Sol_On_Grid   = True_Sol_On_Grid;

        # Initial Conditions
        self.IC_Coords = IC_Coords;
        self.IC_Data   = IC_Data;

        # Periodic BCs
        self.Lower_Bound_Coords = Lower_Bound_Coords;
        self.Upper_Bound_Coords = Upper_Bound_Coords;

        # Training.
        self.Train_Data_Coords  = Train_Data_Coords;
        self.Train_Data_Values  = Train_Data_Values;
        self.Train_Coloc_Coords = Train_Coloc_Coords;

        # Testing.
        self.Test_Data_Coords   = Test_Data_Coords;
        self.Test_Data_Values   = Test_Data_Values;
        self.Test_Coloc_Coords  = Test_Coloc_Coords;



def Data_Loader(
        Mode : str,
        Data_File_Path : str,
        Num_Training_Points : int,
        Num_Testing_Points : int) -> Data_Container:
    """ This function loads data from file and returns it. We make a few
    assumptions about the format of the data. For one, we assume that the .mat
    file contains three fields: x, t, and usol.

    x and t are arrays that are used to construct the grid of points. x and t
    store the set of x, t values in the grid. We assume that the x values are
    uniformly spaced!

    u sol contains the value of the true solution at each gridpoint. Each row of
    usol contains the solution for a fixed position, while each column contains
    the solution for a fixed time.

    We assume the problem has periodic boundary conditions (in space). In
    particular, we assume that row 0 of usol contains the value of the solution
    at the periodic boundary. The last row of usol contains the solution just
    before the periodic boundary. This means that x[0] holds the x coordinate of
    the lower bound of the domain, while the last element of x holds the x
    coordinate just before the upper bound of the domain.

    ----------------------------------------------------------------------------
    Arguments:
    Mode : Either PINNs or Discovery. This controls what data the data loader
    loads and returns. In particular, if we're in "Discovery" mode, then the
    IC and BC members of the Data Container are set to none. If we're in "PINNs"
    mode, then the Training and Testing data/value members are set to None.

    Data_File_Path : A relative path to the data file we want to load.

    Num_Training_Points : The number of training collocation and data points.

    Num_Testing_Points : The number of testing collocation and data points.

    ----------------------------------------------------------------------------
    Returns:
    A Data Container object. See class definition above. """

    data_in = scipy.io.loadmat(Data_File_Path);

    # Fetch spatial, temporal coordinates and the true solution.
    # Note that since we enforce periodic BCs, only one of the spatial bounds
    # is x_points. Because of how Raissi's data saved, this is the lower bound.
    # Thus, x_points will NOT include the upper domain bound.
    x_points = data_in['x'].flatten()[:];
    t_points = data_in['t'].flatten()[:];
    True_Sol_in = np.real(data_in['usol']);

    # Get number of spatial, temporal coordinates.
    n_x = len(x_points);
    n_t = len(t_points);

    # Generate the grid of (x, t) coordinates where we'll evaluate the solution.
    # Each row of these arrays corresponds to a specific position. Each column
    # corresponds to a specific time.
    grid_t_coords, grid_x_coords = np.meshgrid(t_points, x_points);

    # Flatten t_coods, x_coords. use them to generate the test data coodinates.
    flattened_grid_x_coords  = grid_x_coords.flatten()[:, np.newaxis];
    flattened_grid_t_coords  = grid_t_coords.flatten()[:, np.newaxis];
    All_Data_Coords = np.hstack((flattened_grid_x_coords, flattened_grid_t_coords));
        # What's the purpose of [:, np.newaxis]? To make the x, y coordinates
        # into (one column wide) 2d arrays. This is important because of how
        # hstack works. If we feed hstack two 1d arrays, it concatenates them
        # together. Instead, if we feed it two 2d arrays (with the same number
        # of columns) then it concatenates the columns together, which is what
        # we want here.

    All_Data_Values = True_Sol_in.flatten();



    ############################################################################
    # Initial Conditions
    # Since each column of True_Sol_in corresponds to a specific time, the
    # initial condition is just the 0 column of True_Sol_in. We also need the
    # corresponding coordinates.

    # There is an IC coordinate for each possible x value. The corresponding
    # time value for that coordinate is 0.
    IC_Coords = np.zeros((n_x, 2), dtype = np.float);
    IC_Coords[:, 0] = x_points;

    # Since each column of True_Sol_in corresponds to a specific time, the
    # 0 column of True_sol_in holds the initial condition.
    IC_Data = True_Sol_in[:, 0];



    ############################################################################
    # Periodic BC
    # To enforce periodic BCs, for each time, we need the solution to match at
    # the upper and lower spatial bounds. Thus, we need the coordinates of the
    # upper and lower spatial bounds of the domain.

    # x_points only includes the lower spatial bound of the domain. The last
    # element of x_points holds the x value just before the upper bound. Thus,
    # the lower spatial bound is just x_points[0]. The upper spatial bound is
    # x_points[-1] plus the grid spacing (we assume the x values are uniformly
    # spaced).
    x_lower = x_points[0];
    x_upper = x_points[-1] + (x_points[-1] - x_points[-2]);
    x_upper = -x_lower;

    # Now, set up the upper and lower bound coordinates. Let's consider
    # Lower_Bound_Coords. Every coordinate in this array will have the same
    # x coordinate, x_lower. Thus, we initialize an array full of x_lower.
    # We then set the 1 column of this array (the t coordinates) to the
    # set of possible t coordinates (t_points). We do something analagous for
    # Upper_Bound_Coords.
    Lower_Bound_Coords = np.full((n_t, 2), x_lower, dtype = float);
    Upper_Bound_Coords = np.full((n_t, 2), x_upper, dtype = float);
    Lower_Bound_Coords[:, 1] = t_points;
    Upper_Bound_Coords[:, 1] = t_points;



    ############################################################################
    # Training points, values.

    # Randomly select Num_Training_Points coordinate indicies.
    Train_Indicies = np.random.choice(All_Data_Coords.shape[0], Num_Training_Points, replace = False);

    # Select the corresponding collocation points, data points, and data values.
    # Currently, the Coloc and Data coords are the same, though this may change
    # in the future.
    Train_Data_Coords  = torch.from_numpy(All_Data_Coords[Train_Indicies, :]).float();
    Train_Data_Values  = torch.from_numpy(All_Data_Values[Train_Indicies]).float();
    Train_Coloc_Coords = Train_Data_Coords;



    ############################################################################
    # Testing points, values

    # Randomly select Num_Testing_Points coordinate indicies
    Test_Indicies = np.random.choice(All_Data_Coords.shape[0], Num_Testing_Points, replace = False);

    # Note that everything must be of type float32.
    Test_Data_Coords  = torch.from_numpy(All_Data_Coords[Test_Indicies, :]).float();
    Test_Data_Values  = torch.from_numpy(All_Data_Values[Test_Indicies]).float();
    Test_Coloc_Coords = Test_Data_Coords;



    # Package everything into a Data Container object and return. Which objects
    # we return vs which we set to none depends on which Mode we're in.
    # Note that we convert some numpy arrays to tensors (this is almost free,
    # because the tensors share storage with the array).
    if(Mode == "Discovery"):
        # ICs, BCs aren't used in Discovery mode.
        return Data_Container(
                    x_points            = x_points,
                    t_points            = t_points,
                    True_Sol_On_Grid    = True_Sol_in,
                    IC_Coords           = None,
                    IC_Data             = None,
                    Lower_Bound_Coords  = None,
                    Upper_Bound_Coords  = None,
                    Train_Data_Coords   = Train_Data_Coords,
                    Train_Data_Values   = Train_Data_Values,
                    Train_Coloc_Coords  = Train_Coloc_Coords,
                    Test_Data_Coords    = Test_Data_Coords,
                    Test_Data_Values    = Test_Data_Values,
                    Test_Coloc_Coords   = Test_Coloc_Coords);
    elif(Mode == "PINNs"):
        # Testing, Training Coords and Data aren't used in PINNs mode.
        return Data_Container(
                    x_points            = x_points,
                    t_points            = t_points,
                    True_Sol_On_Grid    = True_Sol_in,
                    IC_Coords           = torch.from_numpy(IC_Coords),
                                IC_Data             = torch.from_numpy(IC_Data),
                    Lower_Bound_Coords  = torch.from_numpy(Lower_Bound_Coords),
                    Upper_Bound_Coords  = torch.from_numpy(Upper_Bound_Coords),
                    Train_Data_Coords   = None,
                    Train_Data_Values   = None,
                    Train_Coloc_Coords  = Train_Coloc_Coords,
                    Test_Data_Coords    = None,
                    Test_Data_Values    = None,
                    Test_Coloc_Coords   = Test_Coloc_Coords);
    else:
        print(("Mode is %s while it should be either \"Discovery\" or \"PINNs\"." % Mode));
        print("Something went wrong. Aborting. Thrown by Data_Loader");
        exit();
