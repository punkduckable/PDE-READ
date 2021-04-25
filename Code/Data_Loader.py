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
    x_points = data_in['x'].flatten()[:];
    t_points = data_in['t'].flatten()[:];
    True_in = np.real(data_in['usol']);

    # Get number of spatial, temporal coordinates.
    n_x = len(x_points);
    n_t = len(t_points);

    # Generate the grid of (x, t) coordinates where we'll evaluate the solution.
    # I absolutely hate this function. I find it monstrously unintuitive.
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

    All_Data_Values = True_in.flatten();



    ############################################################################
    # Determine training collocation points, data points, data values.

    # First, randomly select Num_Training_Points indicies.
    Train_Indicies = np.random.choice(All_Data_Coords.shape[0], Num_Training_Points, replace = False);

    # Select the corresponding collocation points, data points, and data values.
    Train_Coloc_Coords = torch.from_numpy(All_Data_Coords[Train_Indicies, :]).float();
    Train_Data_Coords  = Train_Coloc_Coords;
    Train_Data_Values  = torch.from_numpy(All_Data_Values[Train_Indicies]).float();



    ############################################################################
    # Determine training collocation points, data points, data values.

    # First, randomly select Num_Testing_Points indicies
    Test_Indicies = np.random.choice(All_Data_Coords.shape[0], Num_Testing_Points, replace = False);

    # Note that everything must be of type float32.
    Test_Coloc_Coords = torch.from_numpy(All_Data_Coords[Test_Indicies, :]).float();
    Test_Data_Coords  = Test_Coloc_Coords
    Test_Data_Values  = torch.from_numpy(All_Data_Values[Test_Indicies]).float();



    # All done, return.
    return Data_Container(  x_points            = x_points,
                            t_points            = t_points,
                            True_Sol_On_Grid    = True_in,
                            Train_Coloc_Coords  = Train_Coloc_Coords,
                            Train_Data_Coords   = Train_Data_Coords,
                            Train_Data_Values   = Train_Data_Values,
                            Test_Coloc_Coords   = Test_Coloc_Coords,
                            Test_Data_Coords    = Test_Data_Coords,
                            Test_Data_Values    = Test_Data_Values);
