import torch
from typing import Tuple

def Data_Loader(Data_File_Path : str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """ This function loads data from file and returns it. This is basically a
    modified version of part of Raissi's original code.

    ----------------------------------------------------------------------------
    Arguments:
    Data_File_Path : A relative path to the data file we want to load.

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
    print(Data_File_Path);
    exit();
