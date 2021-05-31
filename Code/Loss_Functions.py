import numpy as np;
import torch;

from Network        import Neural_Network;
from PDE_Residual   import PDE_Residual;
from typing         import Tuple;




# Loss from the initial condition.
def IC_Loss(
        u_NN : Neural_Network,
        IC_Coords : torch.Tensor,
        IC_Data : torch.Tensor) -> torch.Tensor:
    """ This function evaluates how well u_NN satisfies the initial condition.
    Specifically, for each point in IC_Coords, we evaluate u_NN. We then
    calculate the square of the difference between this and the corresponding
    true solution in IC_Data. We return the mean of these squared differences.

    ----------------------------------------------------------------------------
    Arguments:
    u_NN : the Neural Network that approximates the solution.

    IC_Coords : The coordinates where we know the true initial condition. This
    should be a two column tensors, each row of which holds an x,t coordinate.

    IC_Data : The value of the initial condition at each point in IC_Coords. If
    IC_Coords has N rows, then this should be an N element tensor. """

    # Pass each IC coordinate through u_NN. This yields a N by 1 tensor whose
    # ith element of this stores the value of the approximate solution at the
    # ith IC Coord. We squeeze out the extra dimension.
    u_approx_batch = u_NN(IC_Coords).squeeze();

    # IC_Data holds the true solution at each point.
    u_true_batch = IC_Data;

    # Calculuate Mean square error.
    Loss =  ((u_true_batch - u_approx_batch)**2).mean();
    return Loss;



# Loss from imposing periodic BCs
def Periodic_BC_Loss(
        u_NN : Neural_Network,
        Lower_Bound_Coords : torch.Tensor,
        Upper_Bound_Coords : torch.Tensor,
        Highest_Order : int) -> torch.Tensor:
    """ This function evaluates how well the learned solution satisfies periodic
    Boundary conditions. Let N = Highest_Order. We require that the solution
    and it's first N derivatives satisify periodic boundary conditions (they
    match at the ends of the spatial domain).

    ----------------------------------------------------------------------------
    Arguments:
    u_NN : The network that approximates the solution to the PDE.

    Lower_Bound_Coords : The x,t coordinates of each point on the lower bound of
    the spatial domain (the x coordinate is always the same, t varies). This
    should be a two column tensors, each row of which holds an x,t coordinate.

    Upper_Bound_Coords : The x,t coordinates of each point on the upper bound of
    the spatial domain (the x coordinate is always the same, t varies). This
    should be a two column tensors, each row of which holds an x,t coordinate.

    Highest_Order : The highest order spatial derivative of the solution that
    we want to impose periodic boundary conditions on. If this is 0, then we
    only apply periodic BCs to the solution itself (not any of its derivatives).

    ----------------------------------------------------------------------------
    Returns :
    A scalar tensor containing the mean square BC error. """

    # Allocate tensors to hold u and its derivatives at each coordinate.
    Num_BC_Points : int = Lower_Bound_Coords.shape[0];
    diu_dxi_upper_batch = torch.empty((Num_BC_Points, Highest_Order+1), dtype = torch.float32);
    diu_dxi_lower_batch = torch.empty((Num_BC_Points, Highest_Order+1), dtype = torch.float32);

    # Evaluate the NN at the upper and lower bound coords. This returns an N by
    # 1 tensor whose ith row holds the value of u at the ith upper or lower
    # coordinate. We squeeze them to get ride of the extra dimension. We also
    # enable gradients because we'll need to evaluate the spatial derivatives of
    # u at each coordinate.
    Upper_Bound_Coords.requires_grad_(True);
    Lower_Bound_Coords.requires_grad_(True);
    diu_dxi_upper_batch[:, 0] = u_NN(Upper_Bound_Coords).squeeze();
    diu_dxi_lower_batch[:, 0] = u_NN(Lower_Bound_Coords).squeeze();

    # Cycle through the derivaitves. For each one, we compute d^ju/dx^j at the
    # two boundaries. To do this, we first compute the gradient of d^ju/dx^j
    # with respect to x, t. The exact way that this works is rather involved
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
        diu_dxi_upper_batch[:, i] = grad_diu_dxi_upper[:, 0];

        grad_diu_dxi_lower = torch.autograd.grad(
                                outputs         = diu_dxi_lower_batch[:, i-1],
                                inputs          = Lower_Bound_Coords,
                                grad_outputs    = torch.ones_like(diu_dxi_lower_batch[:, i-1]),
                                create_graph    = True,
                                retain_graph    = True)[0];
        diu_dxi_lower_batch[:, i] = grad_diu_dxi_lower[:, 0];

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
        u_NN : Neural_Network,
        N_NN : Neural_Network,
        Collocation_Coords : torch.Tensor) -> torch.Tensor:
    """ This function evaluates how well u_NN satisfies the learned PDE at the
    collocation points. For brevity, let u = u_NN and N = N_NN. At each
    collocation point, we compute the following:
                                du/dt + N(u, du/dx, d^2u/dx^2)
    If u actually satisified the learned PDE, then this whould be zero everywhere.
    However, it generally won't be. This function computes the square of the
    quantity above at each Collocation point. We return the mean of these squared
    errors.

    ----------------------------------------------------------------------------
    Arguments:
    u_NN : The neural network that approximates the solution.

    N_NN : The neural network that approximates the PDE.

    Collocation_Coords : a tensor of coordinates of the collocation points. If
    there are N collocation points, then this should be a N x 2 tensor, whose
    ith row holds the x, t coordinate of the ith collocation point.

    ----------------------------------------------------------------------------
    Returns:
    Mean Square Error of the learned PDE at the collocation points. """

    # At each Collocation point, evaluate the square of the residuals
    # du/dt - N(u, du/dx,... ).
    residual_batch = PDE_Residual(
                        u_NN    = u_NN,
                        N_NN    = N_NN,
                        Coords  = Collocation_Coords);

    # Return the mean square residual.
    return (residual_batch ** 2).mean();



# Loss from the training data.
def Data_Loss(
        u_NN : Neural_Network,
        Data_Coords : torch.Tensor,
        Data_Values : torch.Tensor) -> torch.Tensor:
    """ This function evaluates how well the learned solution u satisfies the
    training data. Specifically, for each point ((x_i, t_i), u_i) in
    data, we compute the square of the difference between u_i (the true
    solution at the point (x_i, t_i)) and u(x_i, t_i), where u denotes the
    learned solution. We return the mean of these squared errors. Here the
    phrase "data point" means "a point in the domain at which we know the value
    of the true solution"

    ----------------------------------------------------------------------------
    Arguments:
    u_NN : The neural network that approximates the solution.

    Data_Coords : A tensor of coordinates of the data points. If there are
    N data points, then this should be a N x 2 tensor, whose ith row holds the
    x, t coordinates of the ith data point.

    Data_Values : A tensor containing the value of the true solution at each
    data point. If there are N data points, then this should be an N element
    tesnor whose ith element holds the value of the true solution at the ith
    data point.

    ----------------------------------------------------------------------------
    Returns:
    Mean Square Error between the learned solution and the true solution at
    the data points. """

    # Pass the batch of IC Coordinates through the Neural Network.
    # Note that this will output a N by 1 tensor (where N is the number
    # of coordinates). We need it to be a one dimensional tensor, so we squeeze
    # out the last dimension.
    u_approx_batch = u_NN(Data_Coords).squeeze();

    # Compute Square Error at each coordinate.
    u_true_batch        = Data_Values;
    Square_Error_Batch  = (u_approx_batch - u_true_batch)**2;

    # Return the mean square error.
    return Square_Error_Batch.mean();
