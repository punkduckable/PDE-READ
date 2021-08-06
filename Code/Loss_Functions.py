import numpy as np;
import torch;

from Network        import Neural_Network;
from PDE_Residual   import PDE_Residual;
from typing         import Tuple;



# Loss from the initial condition.
def IC_Loss(
        Sol_NN    : Neural_Network,
        IC_Coords : torch.Tensor,
        IC_Data   : torch.Tensor,
        Data_Type : torch.dtype = torch.float32,
        Device    : torch.device = torch.device('cpu')) -> torch.Tensor:
    """ This function evaluates how well Sol_NN satisfies the initial condition.
    Specifically, for each point (x,t) in IC_Coords, we evaluate Sol_NN and
    calculate |Sol_NN(x, t) - IC(x, t)|^2, where IC(x, t) is the value of the
    true solution at (x, t). We return the mean (over all IC points) of these
    squared differences.

    Note: This function works regardless of how many spatial variables Sol_NN
    depends on.

    ----------------------------------------------------------------------------
    Arguments:

    Sol_NN: The network that approximates the PDE solution.

    IC_Coords: If Sol_NN accepts d spatial variables, then this should be a d+1
    column Tensor, whose ith row holds the t, x_1,... x_d coordinates of a point
    at which we want to enforce the Initial Conditions.

    IC_Data: If IC_Coords has N rows then this should be an N element Tensor
    whose ith element holds the value of the true solutin at the ith IC Coord.

    Data_Type: The data type that all Tensors in Sol_NN use.

    Device: The device that we loaded Sol_NN on.

    ----------------------------------------------------------------------------
    Returns:

    A scalar Tensor containing the mean square IC error. """

    # Pass each IC coordinate through Sol_NN. This yields an N by 1 Tensor whose
    # ith element holds the value of Sol_NN at the ith IC Coord. We squeeze out
    # the extra dimension.
    u_approx_batch = Sol_NN(IC_Coords).squeeze();

    # IC_Data holds the true solution at each point.
    u_true_batch = IC_Data;

    # Calculuate Mean square error.
    Loss =  ((u_true_batch - u_approx_batch) ** 2).mean();
    return Loss;



# Loss from imposing periodic BCs
def Periodic_BC_Loss(
        Sol_NN             : Neural_Network,
        Lower_Bound_Coords : torch.Tensor,
        Upper_Bound_Coords : torch.Tensor,
        Highest_Order      : int,
        Data_Type          : torch.dtype = torch.float32,
        Device             : torch.device = torch.device('cpu')) -> torch.Tensor:
    """ This function evaluates how well Sol_NN satisfies periodic Boundary
    conditions. Let N = Highest_Order. We require that the solution and its
    first N derivatives satisfy periodic boundary conditions (they match at the
    ends of the spatial domain).

    Note: This function only works if Sol_NN is a function of 1 spatial variable.

    ----------------------------------------------------------------------------
    Arguments:

    Sol_NN: The network that approximates the PDE solution.

    Lower_Bound_Coords: This should be a two-column Tensor whose ith row holds
    the t,x coordinates of the ith point on the lower spatial bound.

    Upper_Bound_Coords: This should be a two-column Tensor whose ith row holds
    the t,x coordinates of the ith point on the upper spatial bound.

    Highest_Order: The highest order spatial derivative of Sol_NN for which
    we'll impose periodic boundary conditions. If Highest_Order = 0, we enforce
    periodic BCs only on Sol_NN itself (and none of its derivatives).

    Data_Type: The data type that all Tensors in Sol_NN use.

    Device: The device that we loaded Sol_NN on.

    ----------------------------------------------------------------------------
    Returns:

    A scalar Tensor holding the mean square BC error. """

    # Allocate Tensors to hold Sol_NN and its derivatives at each coordinate.
    Num_BC_Points : int = Lower_Bound_Coords.shape[0];
    diu_dxi_upper_batch = torch.empty((Num_BC_Points, Highest_Order+1), dtype = Data_Type, device = Device);
    diu_dxi_lower_batch = torch.empty((Num_BC_Points, Highest_Order+1), dtype = Data_Type, device = Device);

    # Evaluate Sol_NN at the upper and lower bound coordinates. This returns an
    # N by 1 Tensor whose ith row holds the value of Sol_NN at the ith upper or
    # lower coordinate. We squeeze them to get ride of the extra dimension. We
    # also enable gradients because we'll need toevaluate the spatial
    # derivatives of Sol_NN at each coordinate.
    Upper_Bound_Coords.requires_grad_(True);
    Lower_Bound_Coords.requires_grad_(True);
    diu_dxi_upper_batch[:, 0] = Sol_NN(Upper_Bound_Coords).squeeze();
    diu_dxi_lower_batch[:, 0] = Sol_NN(Lower_Bound_Coords).squeeze();

    # Cycle through the derivatives. For each one, we compute d^ju/dx^j at the
    # upper and lower boundaries. To do this, we first compute the gradient of
    # d^ju/dx^j with respect to t, x. How exactly this works exactly is fairly
    # involved. You can read about it in my extensive comment in
    # Evaluate_Sol_Derivatives (which does the same thing for a different loss
    # function). We create a graph for the new derivative, and retain the graph
    # for the old one so we can differentiate the loss function!
    for i in range(1, Highest_Order + 1):
        grad_diu_dxi_upper = torch.autograd.grad(
                                outputs         = diu_dxi_upper_batch[:, i-1],
                                inputs          = Upper_Bound_Coords,
                                grad_outputs    = torch.ones_like(diu_dxi_upper_batch[:, i-1]),
                                create_graph    = True,
                                retain_graph    = True)[0];
        diu_dxi_upper_batch[:, i] = grad_diu_dxi_upper[:, 1];

        grad_diu_dxi_lower = torch.autograd.grad(
                                outputs         = diu_dxi_lower_batch[:, i-1],
                                inputs          = Lower_Bound_Coords,
                                grad_outputs    = torch.ones_like(diu_dxi_lower_batch[:, i-1]),
                                create_graph    = True,
                                retain_graph    = True)[0];
        diu_dxi_lower_batch[:, i] = grad_diu_dxi_lower[:, 1];

    # Now let's compute the BC error at each BC coordinate. A lot is going on
    # here. First, we compute the element-wise difference of grad_diu_dxi_upper
    # and grad_diu_dxi_lower. Next, we compute the pointwise square of this
    # Tensor and sum the columns (sum along the rows). Let xt_H_i denote
    # Upper_Bound_Coords[i] and xt_L_i denote Lower_Bound_Coords[i]. The ith
    # component of the resulting Tensor holds
    #   sum_{i = 0}^{Highest_Order} |d^iu/dx^i(xt_H_i) - d^iu/dx^i(xt_L_i)|^2
    Square_BC_Errors_batch = ((diu_dxi_upper_batch - diu_dxi_lower_batch) ** 2).sum(dim = 1);

    # The loss is then the mean of the Square BC Errors.
    return Square_BC_Errors_batch.mean();



# Loss from enforcing the PDE at the collocation points.
def Collocation_Loss(
        Sol_NN             : Neural_Network,
        PDE_NN             : Neural_Network,
        Collocation_Coords : torch.Tensor,
        Data_Type          : torch.dtype = torch.float32,
        Device             : torch.device = torch.device('cpu')) -> torch.Tensor:
    """ This function evaluates how well Sol_NN satisfies the learned PDE at the
    collocation points. For brevity, let u = Sol_NN and N = PDE_NN. At each
    collocation point, we compute the following:
                                du/dt - N(u, du/dx, d^2u/dx^2)
    If Sol_NN satisfies the learned PDE, then this quantity will be zero
    everywhere. However, it generally won't be. This function computes the
    square of the quantity above at each Collocation point. We return the mean
    of these squared errors.

    Note: This function only works if Sol_NN is a function of 1 spatial variable.

    ----------------------------------------------------------------------------
    Arguments:

    Sol_NN: The network that approximates the PDE solution.

    PDE_NN: The network that approximates the PDE.

    Collocation_Coords: This should be a 2 column Tensor whose ith row holds the
    t, x coordinates of the ith collocation point.

    Data_Type: The data type that all Tensors in Sol_NN and PDE_NN use.

    Device: The device that we loaded Sol_NN and PDE_NN on.

    ----------------------------------------------------------------------------
    Returns:

    A scalar Tensor containing the mean square collocation error. """

    # At each Collocation point, evaluate the square of the residuals
    # du/dt - N(u, du/dx,... ).
    residual_batch = PDE_Residual(
                        Sol_NN    = Sol_NN,
                        PDE_NN    = PDE_NN,
                        Coords    = Collocation_Coords,
                        Data_Type = Data_Type,
                        Device    = Device);

    # Return the mean square residual.
    return (residual_batch ** 2).mean();



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
