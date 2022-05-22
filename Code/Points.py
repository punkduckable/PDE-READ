import numpy;
import torch;
import random;



def Generate_Points(
        Bounds     : numpy.array,
        Num_Points : int,
        Data_Type  : torch.dtype,
        Device     : torch.device = torch.device('cpu')) -> torch.Tensor:
    """ This function generates a two-dimensional tensor, each row of which
    holds a randomly generated coordinate that lies in the rectangle defined by
    Bounds.

    ----------------------------------------------------------------------------
    Arguments:

    Bounds: A two-column tensor. Whose ith row contains the lower and upper
    bounds of the ith sub-rectangle of the rectangle.

    Num_Points: The number of points we want to generate.

    Data_Type: The data type used for the coords. Should be torch.float64
    (double precision) or torch.float32 (single precision).
    
    Device: The device you want the Point tensor to be stored on.

    ----------------------------------------------------------------------------
    Returns:

    A Num_Points row tensor, each row of which contains a randomly generated
    coordinate in the rectangle specified by Bounds. Suppose that
            Bounds = [[a_1, b_1], ... , [a_n, b_n]]
    Then the ith row of the returned tensor contains a coordinate that lies
    within [a_1, b_1]x...x[a_n, b_n]. """


    # First, determine the number of dimensions. This is just the number of rows
    # in Bounds.
    Num_Dim : int = Bounds.shape[0];

    # Check that the Bounds are valid.
    for j in range(Num_Dim):
        assert(Bounds[j, 0] <= Bounds[j, 1]);

    # Make a tensor to hold all the points.
    Points = torch.empty((Num_Points, Num_Dim),
                          dtype  = Data_Type,
                          device = Device);

    # Populate the coordinates in Points, one coordinate at a time.
    for j in range(Num_Dim):
        # Get the upper and lower bounds for the jth coordinate.
        Lower_Bound : float = Bounds[j, 0];
        Upper_Bound : float = Bounds[j, 1];

        # Cycle through the points.
        for i in range(Num_Points):
            Points[i, j] = random.uniform(Lower_Bound, Upper_Bound);

    return Points;
