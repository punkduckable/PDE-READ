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

    # What we do next depends on which mode we're in.
    if  (Mode == "PINNs"):
        # If we're in PINN's mode, then the DataSet should contain IC coords/
        # targets, as well as upper/lower boundary coords. Fetch them.
        IC_Coords   : numpy.ndarray = DataSet["IC_Inputs"];
        IC_Data     : numpy.ndarray = DataSet["IC_Targets"];

        Lower_Bound_Coords : numpy.ndarray = DataSet["Lower_Bound_Inputs"];
        Upper_Bound_Coords : numpy.ndarray = DataSet["Upper_Bound_Inputs"];

        # Add the tensor version of these items to the container
        Container.IC_Coords        = torch.from_numpy(IC_Coords).to(device = Device);
        Container.IC_Data          = torch.from_numpy(IC_Data)  .to(device = Device);

        Container.Lower_Bound_Coords = torch.from_numpy(Lower_Bound_Coords).to(device = Device);
        Container.Upper_Bound_Coords = torch.from_numpy(Upper_Bound_Coords).to(device = Device);

        # Determine the size of the domain (for Collocation point generation).
        # The IC_Coords, Lower_Bound_Coords, and Upper_Bound_Inputs contain
        # the relevant information.
        t_min : float = IC_Coords[0, 0];                # t component of first IC coord.
        t_max : float = Lower_Bound_Coords[-1, 0];      # t component of last boundary coord

        x_min : float = Lower_Bound_Coords[0, 1];       # x component of first lower BC coord.
        x_0     : float = IC_Coords[0, 1];              # x component of first IC coord.
        x_1     : float = IC_Coords[1, 1];              # x component of second IC coord.
        dx      : float = abs(x_1 - x_0);               # distanct between successive IC coords.
        x_max : float = Upper_Bound_Coords[0, 1] + dx;  # x componnet of first upper BC coord.

        # Store these bounds in the Container.
        Container.Dim_Lower_Bounds = numpy.array((t_min, x_min), dtype = numpy.float32);
        Container.Dim_Upper_Bounds = numpy.array((t_max, x_max), dtype = numpy.float32);

    elif(Mode == "Discovery" or Mode == "Extraction"):
        # If we're in Discovery mode, then we need Testing/Training Data
        # coordinates and values. Fetch them.
        Train_Data_Coords   : numpy.ndarray = DataSet["Train_Inputs"];
        Train_Data_Values   : numpy.ndarray = DataSet["Train_Targets"];

        Test_Data_Coords    : numpy.ndarray = DataSet["Test_Inputs"];
        Test_Data_Values    : numpy.ndarray = DataSet["Test_Targets"];

        # Add the tensor version of these items to the Container.
        Container.Train_Data_Coords = torch.from_numpy(Train_Data_Coords).to(device = Device);
        Container.Train_Data_Values = torch.from_numpy(Train_Data_Values).to(device = Device);

        Container.Test_Data_Coords  = torch.from_numpy(Test_Data_Coords).to(device = Device);
        Container.Test_Data_Values  = torch.from_numpy(Test_Data_Values).to(device = Device);

        # If we're in Discovery or Extraction mode, then we need to know the
        # bounds of the problem domain. We use these bounds to select
        # collocation/extraction points. To identify the bounds, we find the
        # max/min x/t coordinates - x_min, x_max, t_min, and t_max - in the
        # training set. We assume the problem domain is [t_min, t_max] x
        # [x_min, x_max].
        t_min : float = Train_Data_Coords[0, 0];        # t component of first training point.
        t_max : float = Train_Data_Coords[0, 0];        # t component of first training point.

        x_min : float = Train_Data_Coords[0, 1];        # x component of first training point.
        x_max : float = Train_Data_Coords[0, 1];        # x component of first training point.

        N : int = Train_Data_Coords.shape[0];
        for i in range(N):
            t_i : float = Train_Data_Coords[i, 0];
            x_i : float = Train_Data_Coords[i, 1];

            if(t_i < t_min):
                t_min = t_i;
            if(t_i > t_max):
                t_max = t_i;
            if(x_i < x_min):
                x_min = x_i;
            if(x_i > x_max):
                x_max = x_i;

        # Store these bounds in the Container.
        Container.Dim_Lower_Bounds = numpy.array((t_min, x_min), dtype = numpy.float32);
        Container.Dim_Upper_Bounds = numpy.array((t_max, x_max), dtype = numpy.float32);

    # The container is now full. Return it!
    return Container;



def Generate_Random_Coords(
        Dim_Lower_Bounds    : numpy.array,
        Dim_Upper_Bounds    : numpy.array,
        Num_Points          : int,
        Data_Type           : torch.dtype = torch.float32,
        Device              : torch.device = torch.device('cpu')) -> torch.Tensor:
    """ This function generates a collection of random points within the box
    specified by Dim_Lower_Bounds and Dim_Upper_Bounds.

    ----------------------------------------------------------------------------
    Arguments:

    dim_lower_bounds: If we want to generate points in R^d, then this should be
    a d element array whose kth element stores the lower bound for the kth
    variable.

    dim_upper_bounds: same as dim_lower_bounds but for upper bounds.

    num_Points: The number of points we want to generate.

    Data_Type: The data type used for the coords. Should be torch.float64
    (double precision) or torch.float32 (single precision).

    ----------------------------------------------------------------------------
    Returns:

    A num_Points by d array (where d is the dimension of the space in which the
    points live) whose ith row contains the coordinates of the ith point. """

    # Declare coords array
    d = Dim_Lower_Bounds.size;
    Coords = torch.empty((Num_Points, d), dtype = Data_Type, device = Device);

    # Populate the coordinates with random values.
    for i in range(Num_Points):
        for k in range(d):
            Coords[i, k] = random.uniform(Dim_Lower_Bounds[k], Dim_Upper_Bounds[k]);

    return Coords;
