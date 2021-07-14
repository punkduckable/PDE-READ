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
    Specifically, for each point in IC_Coords, we evaluate Sol_NN. We then
    calculate the square of the difference between this and the corresponding
    true solution in IC_Data. We return the mean of these squared differences.

    Note: This function works regardless of how many spatial variables u depends
    on.

    ----------------------------------------------------------------------------
    Arguments:

    Sol_NN: The network that approximates the PDE solution.

    IC_Coords: The coordinates where we know the true initial condition. If u
    accepts d spatial variables, then this should be a d+1 column tensor, whose
    ith row holds the t, x_1,... x_d coordinates of a Point we want to enforce
    the initial condition.

    IC_Data: The value of the initial condition at each point in IC_Coords. If
    IC_Coords has N rows, then this should be an N element tensor.

    Data_Type: The data type that all tensors use. All tensors in Sol_NN should
    use this data type.

    Device: The device that Sol_NN is loaded on.

    ----------------------------------------------------------------------------
    Returns:

    A scalar tensor containing the mean square IC error. """

    # Pass each IC coordinate through Sol_NN. This yields a N by 1 tensor whose
    # ith element of this stores the value of the approximate solution at the
    # ith IC Coord. We squeeze out the extra dimension.
    u_approx_batch = Sol_NN(IC_Coords).squeeze();

    # IC_Data holds the true solution at each point.
    u_true_batch = IC_Data;

    # Calculuate Mean square error.
    Loss =  ((u_true_batch - u_approx_batch)**2).mean();
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
    first N derivatives satisify periodic boundary conditions (they match at the
    ends of the spatial domain).

    Note: this function only works if the solution is a function of 1 spatial
    variable.

    ----------------------------------------------------------------------------
    Arguments:

    Sol_NN: The network that approximates the PDE solution.

    Lower_Bound_Coords: The t,x coordinates of each point on the lower bound of
    the spatial domain. This should be a 2 column tensor whose ith row holds the
    t,x coordinates of the ith point where we'll enforce the boundary condition.

    Upper_Bound_Coords: The t,X coordinates of each point on the upper bound of
    the spatial domain. This should be a 2 column tensor whose ith row holds the
    t,x coordinates of the ith point where we'll enforce the boundary condition.

    Highest_Order: The highest order spatial derivative of the solution that
    we want to impose periodic boundary conditions on. If this is 0, then we
    only apply periodic BCs to the solution itself (not any of its derivatives).

    Data_Type: The data type that all tensors use. All tensors in Sol_NN should
    use this data type.

    Device: The device that Sol_NN is loaded on.

    ----------------------------------------------------------------------------
    Returns:

    A scalar tensor containing the mean square BC error. """

    # Allocate tensors to hold u and its derivatives at each coordinate.
    Num_BC_Points : int = Lower_Bound_Coords.shape[0];
    diu_dxi_upper_batch = torch.empty((Num_BC_Points, Highest_Order+1), dtype = Data_Type, device = Device);
    diu_dxi_lower_batch = torch.empty((Num_BC_Points, Highest_Order+1), dtype = Data_Type, device = Device);

    # Evaluate the NN at the upper and lower bound coords. This returns an N by
    # 1 tensor whose ith row holds the value of u at the ith upper or lower
    # coordinate. We squeeze them to get ride of the extra dimension. We also
    # enable gradients because we'll need to evaluate the spatial derivatives of
    # u at each coordinate.
    Upper_Bound_Coords.requires_grad_(True);
    Lower_Bound_Coords.requires_grad_(True);
    diu_dxi_upper_batch[:, 0] = Sol_NN(Upper_Bound_Coords).squeeze();
    diu_dxi_lower_batch[:, 0] = Sol_NN(Lower_Bound_Coords).squeeze();

    # Cycle through the derivaitves. For each one, we compute d^ju/dx^j at the
    # two boundaries. To do this, we first compute the gradient of d^ju/dx^j
    # with respect to t, x. The exact way that this works is rather involved
    # read my extensive comment in the PDE_Residual function (which basically
    # does the same thing for a different loss function).
    # We create a graph for the new derivative, and retain the graph for the
    # old one because we need to be able to differentiate the loss function!
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

    # Now let's compute the BC error at each BC coordinate. There's a lot going
    # on here. First, we compute the element-wise difference of
    # grad_diu_dxi_upper and grad_diu_dxi_lower. Next, we compute the pointwise
    # square of this tensor and sum along the rows. Let xt_H_i denote
    # Upper_Bound_Coords[i] and xt_L_i denote Lower_Bound_Coords[i]. The
    # ith component of the resulting tensor holds
    #   sum_{i = 0}^{Highest Order} |d^iu/dx^i(xt_H_i) - d^iu/dx^i(xt_L_i)|^2
    Square_BC_Errors_batch = ((diu_dxi_upper_batch - diu_dxi_lower_batch)**2).sum(dim = 1);

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
                                du/dt + N(u, du/dx, d^2u/dx^2)
    If u actually satisified the learned PDE, then this whould be zero everywhere.
    However, it generally won't be. This function computes the square of the
    quantity above at each Collocation point. We return the mean of these squared
    errors.

    Note: this function only works is u is a function of 1 spatial variable.

    ----------------------------------------------------------------------------
    Arguments:

    Sol_NN: The network that approximates the PDE solution.

    PDE_NN: The network that approximates the PDE.

    Collocation_Coords: a tensor of coordinates of the collocation points. This
    should be a 2 column tensor whose ith row holds the t, x coordinates of the
    ith collocation point.

    Data_Type: The data type that all tensors use. All tensors in Sol_NN and
    PDE_NN should use this data type.

    Device: The device that Sol_NN and PDE_NN are loaded on.

    ----------------------------------------------------------------------------
    Returns:

    A scalar tensor containing the mean square collocation error. """

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
    """ This function evaluates how well the learned solution u satisfies the
    training data. Specifically, for each point ((t_i, X_i), u_i) in
    data, we compute the square of the difference between u_i (the true
    solution at the point (t_i, X_i)) and u(t_i, X_i), where u denotes the
    learned solution. We return the mean of these squared errors. Here the
    phrase "data point" means "a point in the domain at which we know the value
    of the true solution"

    Note: This function works regardless of how many spatial variables u depends
    on.

    ----------------------------------------------------------------------------
    Arguments:

    Sol_NN: The network that approximates the PDE solution.

    Data_Coords: A tensor of coordinates of the data points. If u takes d
    spatial variables, then this should be a d+1 column tensor whose ith row
    holds the t, x_1,... x_d coordinates of the ith data point.

    Data_Values: A tensor containing the value of the true solution at each
    data point. If there are N data points, then this should be an N element
    tesnor whose ith element holds the value of the true solution at the ith
    data point.

    Data_Type: The data type that all tensors use. All tensors in Sol_NN should
    use this data type.

    Device: The device that Sol_NN and PDE_NN are loaded on.

    ----------------------------------------------------------------------------
    Returns:

    A scalar tensor containing the mean square error between the learned and
    true solutions at the data points. """

    # Pass the batch of IC Coordinates through the Neural Network.
    # Note that this will output a N by 1 tensor (where N is the number
    # of coordinates). We need it to be a one dimensional tensor, so we squeeze
    # out the last dimension.
    u_approx_batch = Sol_NN(Data_Coords).squeeze();

    # Compute the square error at each coordinate.
    u_true_batch        = Data_Values;
    Square_Error_Batch  = (u_approx_batch - u_true_batch)**2;

    # Return the mean square error.
    return Square_Error_Batch.mean();
