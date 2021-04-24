import torch;
import numpy as np;
import scipy.io;

from typing import Tuple;



def Data_Loader(Data_File_Path : str, Num_Training_Points : int, Num_Testing_Points : int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """ This function loads data from file and returns it. This is basically a
    modified version of part of Raissi's original code.

    ----------------------------------------------------------------------------
    Arguments:
    Data_File_Path : A relative path to the data file we want to load.

    Num_Training_Points : The number of training collocation and data points.

    Num_Testing_Points : The number of testing collocation and data points.

    ----------------------------------------------------------------------------
    Returns:
    A tuple of 6 elements. We shall denote the ith return argument by R_i. That
    is, this function returns (R_1, R_2, R_3, R_4, R_5, R_6). Here's what
    R_1-R_6 contain:
        R_1 : Coordinates of the collocation points for training
        R_2 : Coordinates of the data points for training
        R_3 : Value of the true solution at each point of R_2.
        R_4 : Coordinates of the collocation points for testing
        R_5 : Coordinates of the data points for testing
        R_6 : Value of the true solution at each point of R_5.
    """

    # First, load the file
    data_in = scipy.io.loadmat(Data_File_Path);

    # Fetch spatial, temporal coordinates and the exact solution.
    t_in = data_in['t'].flatten()[:];
    x_in = data_in['x'].flatten()[:];
    Exact_in = np.real(data_in['usol']);

    # Generate the grid of (x, t) coordinates where we'll evaluate the solution.
    t_coords, x_coords = np.meshgrid(t_in, x_in);

    # Flatten t_coods, x_coords. use them to generate the test data coodinates.
    t_coords_flat = t_coords.flatten()[:, np.newaxis];
    x_coords_flat = x_coords.flatten()[:, np.newaxis];
    All_Data_Coords = np.hstack((t_coords_flat, x_coords_flat));
        # What's the purpose of [:, np.newaxis]? To make t_in, x_in
        # into (one column wide) 2d arrays. If there are N_t times in data['t'],
        # then t_in will be a Nx1 array. This is important because of how
        # hstack works. If we feed hstack two 1d arrays, it concatenates them
        # together. Instead, if we feed it two 2d arrays (with the same number
        # of columns) then it concatenates the columns together, which is what
        # we want here.

    All_Data_Values = Exact_in.flatten();



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
    return (Train_Coloc_Coords, Train_Data_Coords, Train_Data_Values, Test_Coloc_Coords, Test_Data_Coords, Test_Data_Values);
