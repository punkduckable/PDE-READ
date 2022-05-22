import numpy as np;
import torch;

from Network        import Neural_Network;
from PDE_Residual   import PDE_Residual;
from typing         import Tuple;



# Loss from enforcing the PDE at the collocation points.
def Collocation_Loss(
        Sol_NN                      : Neural_Network,
        PDE_NN                      : Neural_Network,
        Time_Derivative_Order       : int,
        Spatial_Derivative_Order    : int,
        Collocation_Coords          : torch.Tensor,
        Data_Type                   : torch.dtype = torch.float32,
        Device                      : torch.device = torch.device('cpu')) -> torch.Tensor:
    """ This function evaluates how well Sol_NN satisfies the learned PDE at the
    collocation points. For brevity, let u = Sol_NN and N = PDE_NN. At each
    collocation point, we compute the following:
                    D_t^m U - N(u, D_x U, D_x^2 U, ... D_x^n U)
    where m = Time_Derivative_Order, and n = Spatial_Derivative_Order.
    If Sol_NN satisfies the learned PDE, then this quantity will be zero
    everywhere. However, it generally won't be. This function computes the
    square of the quantity above at each Collocation point. We return the mean
    of these squared errors.

    Note: This function only works if Sol_NN is a function of 1 spatial variable.

    ----------------------------------------------------------------------------
    Arguments:

    Sol_NN: The network that approximates the PDE solution.

    PDE_NN: The network that approximates the PDE.

    Time_Derivative_Order: The order of the time derivative in the PDE we're
    trying to solve.

    Collocation_Coords: This should be a 2 column Tensor whose ith row holds the
    t, x coordinates of the ith collocation point.

    Data_Type: The data type that all Tensors in Sol_NN and PDE_NN use.

    Device: The device that we loaded Sol_NN and PDE_NN on.

    ----------------------------------------------------------------------------
    Returns:

    A scalar Tensor containing the mean square collocation error. """

    # At each Collocation point, evaluate the square of the residuals
    # du/dt - N(u, du/dx,... ).
    Residual : torch.Tensor = PDE_Residual(
                                Sol_NN                      = Sol_NN,
                                PDE_NN                      = PDE_NN,
                                Time_Derivative_Order       = Time_Derivative_Order,
                                Spatial_Derivative_Order    = Spatial_Derivative_Order,
                                Coords                      = Collocation_Coords,
                                Data_Type                   = Data_Type,
                                Device                      = Device);

    # Return the mean square residual.
    return (Residual ** 2).mean();



# Loss from the training data.
def Data_Loss(
        Sol_NN        : Neural_Network,
        Data_Coords : torch.Tensor,
        Data_Values : torch.Tensor,
        Data_Type   : torch.dtype = torch.float32,
        Device      : torch.device = torch.device('cpu')) -> torch.Tensor:
    """ This function evaluates how well Sol_NN satisfies the training data.
    Specifically, for Data point (t_i, X_i), we compute the square of the
    difference between u_i (the true solution at (t_i, X_i)) and
    Sol_NN(t_i, X_i). We return the mean of these squared errors. Here the
    phrase "data point" means "a point in the domain at which we know the value
    of the true solution."

    Note: This function works regardless of how many spatial variables Sol_NN
    depends on.

    ----------------------------------------------------------------------------
    Arguments:

    Sol_NN: The network that approximates the PDE solution.

    Data_Coords: If Sol_NN takes d spatial variables, then this should be a d+1
    column Tensor whose ith row holds the t, x_1,... x_d coordinates of the ith
    data point.

    Data_Values: A eTnsor containing the value of the true solution at each
    data point. If there are N data points, then this should be an N element
    Tensor whose ith element holds the value of the true solution at the ith
    data point.

    Data_Type: The data type that all Tensors use. All Tensors in Sol_NN should
    use this data type.

    Device: The device that we loaded Sol_NN on.

    ----------------------------------------------------------------------------
    Returns:

    A scalar Tensor containing the mean square error between the learned and
    true solutions at the data points. """

    # Pass the batch of IC Coordinates through the Neural Network. Note that
    # this will output an N by 1 Tensor (where N is the number of  coordinates).
    # We need it to be a one-dimensional Tensor, so we squeeze out the extra
    # dimension.
    u_approx_batch = Sol_NN(Data_Coords).squeeze();

    # Compute the square error at each coordinate.
    u_true_batch        = Data_Values;
    Square_Error_Batch  = (u_approx_batch - u_true_batch) ** 2;

    # Return the mean square error.
    return Square_Error_Batch.mean();
