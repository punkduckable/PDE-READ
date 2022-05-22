import numpy;
import torch;
import random;



class Data_Container:
    pass;



def Data_Loader(DataSet_Name   : str,
                Device         : torch.device,
                Mode           : str):
    """ This function loads a DataSet from file, converts it contents to a torch
    Tensor, and returns the result.

    Note that if we're in Discovery mode, then we assume that the DataSet file
    contains Test/Train_Inputs, and Test/Train_Targets. By contrast, if we're
    in PINNs mode, we assume that the DataSet file contains IC_Inputs/Targets,
    and Upper/Lower_Bound_Coords. Here, Upper_Bound_Coords and Lower_Bound_Coords
    are lists of the form {(t_0, x_L), ... , (t_N, x_L)} and {(t_0, x_H), ... ,
    (t_M, x_H)}, where x_L and x_H represent the upper and lower edges of the
    spatial domain. We assume the coordinates in IC_Inputs are uniformly spaced,
    with spacing dx, and that the periodic boundary occurs dx beyond the x
    coordinate of the last IC_Input.

    Note: This function is currently hardcoded to work with data involving 1
    spatial dimension.

    ----------------------------------------------------------------------------
    Arguments:

    DataSet_Name : The name of a file in Data/DataSets (without the .npz
    extension). We load the DataSet in this file.

    Device : The device we're running training on.

    Mode : Which mode we're running in (PINNs or Discovery).

    ----------------------------------------------------------------------------
    Returns:

    A Data Container object. What's in that container depends on which mode
    we're in. """

    # Load the DataSet.
    DataSet_Path    = "../Data/DataSets/" + DataSet_Name + ".npz";
    DataSet         = numpy.load(DataSet_Path);

    # Make the Container.
    Container = Data_Container();

    # First, fetch the training/testing inputs and targets.
    Train_Inputs    : numpy.ndarray = DataSet["Train_Inputs"];
    Train_Targets   : numpy.ndarray = DataSet["Train_Targets"];

    Test_Inputs     : numpy.ndarray = DataSet["Test_Inputs"];
    Test_Targets    : numpy.ndarray = DataSet["Test_Targets"];

    # Convert these to tensors and add them to the container.
    Container.Train_Inputs  = torch.from_numpy(Train_Inputs).to(device = Device);
    Container.Train_Targets = torch.from_numpy(Train_Targets).to(device = Device);

    Container.Test_Inputs  = torch.from_numpy(Test_Inputs).to(device = Device);
    Container.Test_Targets = torch.from_numpy(Test_Targets).to(device = Device);

    # Finally, fetch the Input Bounds array.
    Container.Input_Bounds = DataSet["Input_Bounds"];

    # The container is now full. Return it!
    return Container;
