import torch;
import numpy as np;
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
                    IC_Coords           : np.array,
                    IC_Data             : np.array,
                    Lower_Bound_Coords  : np.array,
                    Upper_Bound_Coords  : np.array,
                    True_Sol_On_Grid    : np.array,
                    Train_Coloc_Coords  : torch.Tensor,
                    Train_Data_Coords   : torch.Tensor,
                    Train_Data_Values   : torch.Tensor,
                    Test_Coloc_Coords   : torch.Tensor,
                    Test_Data_Coords    : torch.Tensor,
                    Test_Data_Values    : torch.Tensor):

        # For plotting.
        self.x_points           = x_points;
        self.t_points           = t_points;
        self.True_Sol_On_Grid   = True_Sol_On_Grid;

        # For Initial Conditions
        self.IC_Coords = IC_Coords;
        self.IC_Data   = IC_Data;

        # For Periodic BCs
        self.Lower_Bound_Coords = Lower_Bound_Coords;
        self.Upper_Bound_Coords = Upper_Bound_Coords;

        # For training.
        self.Train_Coloc_Coords = Train_Coloc_Coords;
        self.Train_Data_Coords  = Train_Data_Coords;
        self.Train_Data_Values  = Train_Data_Values;

        # For testing.
        self.Test_Coloc_Coords  = Test_Coloc_Coords;
        self.Test_Data_Coords   = Test_Data_Coords;
        self.Test_Data_Values   = Test_Data_Values;



def Data_Loader(Data_File_Path : str, Num_Training_Points : int, Num_Testing_Points : int) -> Data_Container:
    """ This function loads data from file and returns it. This is basically a
    modified version of part of Raissi's original code.

    ----------------------------------------------------------------------------
    Arguments:
    Data_File_Path : A relative path to the data file we want to load.

    Num_Training_Points : The number of training collocation and data points.

    Num_Testing_Points : The number of testing collocation and data points.

    ----------------------------------------------------------------------------
    Returns:
    A Data Container object. See class definition above. """

    # First, load the file
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
    # grid_t_coords and grid_x_coords are 2d numpy arrays. Each row of these
    # arrays corresponds to a specific position. Each column corresponds to a
    # specific time.
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
    # Determine the initial conditions
    # Since each column of True_Sol_in corresponds to a specific time, the
    # initial condition is just the 0 column of True_Sol_in. We also need the
    # corresponding coordinates.

    # Generate IC coordinates. There is an IC coordinate for each possible
    # x value. The corresponding time value for that coordinate is 0.
    IC_Coords = np.zeros((n_x, 2), dtype = np.float);
    IC_Coords[:, 0] = x_points;

    # Since each column of True_Sol_in corresponds to a specific time, the
    # initial condition is just the 0 column of True_Sol_in.
    IC_Data = True_Sol_in[:, 0];



    ############################################################################
    # Determine coordinates to enforce periodic BC.
    # To enforce periodic BCs, for each time, we need the solution to match at
    # the upper and lower spatial bounds. Thus, we need the coordinates of the
    # upper and lower spatial bounds of the domain.

    # Recall that x_points only includes the lower spatial bound of the
    # domain (the upper spatial bound is not the last element of this array).
    # We can get the lower spatial bound from x_points. The upper spatial bound
    # is then the negative of this value (we assume a symmetric spatial domain).
    x_lower = x_points[0];
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
    # Determine training collocation points, data points, data values.

    # First, randomly select Num_Training_Points indicies.
    Train_Indicies = np.random.choice(All_Data_Coords.shape[0], Num_Training_Points, replace = False);

    # Select the corresponding collocation points, data points, and data values.
    Train_Coloc_Coords = torch.from_numpy(All_Data_Coords[Train_Indicies, :]).float();
    Train_Data_Coords  = Train_Coloc_Coords;
    Train_Data_Values  = torch.from_numpy(All_Data_Values[Train_Indicies]).float();



    ############################################################################
    # Determine testing collocation points, data points, data values.

    # First, randomly select Num_Testing_Points indicies
    Test_Indicies = np.random.choice(All_Data_Coords.shape[0], Num_Testing_Points, replace = False);

    # Note that everything must be of type float32.
    Test_Coloc_Coords = torch.from_numpy(All_Data_Coords[Test_Indicies, :]).float();
    Test_Data_Coords  = Test_Coloc_Coords;
    Test_Data_Values  = torch.from_numpy(All_Data_Values[Test_Indicies]).float();



    # All done. Package everything into a Data Container object and return.
    return Data_Container(  x_points            = x_points,
                            t_points            = t_points,
                            IC_Coords           = IC_Coords,
                            IC_Data             = IC_Data,
                            Lower_Bound_Coords  = Lower_Bound_Coords,
                            Upper_Bound_Coords  = Upper_Bound_Coords,
                            True_Sol_On_Grid    = True_Sol_in,
                            Train_Coloc_Coords  = Train_Coloc_Coords,
                            Train_Data_Coords   = Train_Data_Coords,
                            Train_Data_Values   = Train_Data_Values,
                            Test_Coloc_Coords   = Test_Coloc_Coords,
                            Test_Data_Coords    = Test_Data_Coords,
                            Test_Data_Values    = Test_Data_Values);
