import numpy as np;
import torch;

from typing import Tuple;
from Network import Neural_Network;



def Evaluate_Sol_Derivatives(
        Sol_NN          : Neural_Network,
        num_derivatives : int,
        Coords          : torch.Tensor,
        Data_Type       : torch.dtype = torch.float32,
        Device          : torch.device = torch.device('cpu')) -> Tuple[torch.Tensor, torch.Tensor]:
    """ This function evaluates u, du/dt, and d^i u/dx^i (for i = 1,2... ) at
    each coordinate in Coords.

    Note: This function only works if Sol_NN is a function of 1 spatial variable.

    ----------------------------------------------------------------------------
    Arguments:

    Sol_NN: The network that approximates the PDE solution.

    num_derivatives: The number of spatial derivatives of Sol_NN we need to
    evaluate.

    Coords: A two-column Tensor whose ith row holds the t, x coordinates of the
    ith point we'll evaluate Sol_NN and its derivatives at.

    Data_Type: The data type that all Tensors in Sol_NN use.

    Device: The device that Sol_NN is loaded on.

    ----------------------------------------------------------------------------
    Returns:

    This returns a two-element Tuple! For brevity in what follows, let
    u = Sol_NN. If Coords is an M by 2 Tensor, then the first return argument
    is an M element Tensor whose ith element holds the value of du/dt at the ith
    coordinate. If PDE_NN accepts N arguments, then the second return variable
    is an M by N Tensor whose i,j element holds the value of d^ju/dx^j at the
    ith coordinate. """

    # We need to evaluate derivatives, so set Requires Grad to true.
    Coords.requires_grad_(True);

    # Determine how many derivatives of Sol_NN we'll need to evaluate the PDE.
    # Remember that PDE_NN is a function of u, du/dx, d^2u/dx^2,
    # d^(n-1)u/dx^(n-1), where n is the number of inputs that PDE_NN accepts.
    # Once we know this, and the number of Collocation points, we initialize a
    # Tensor to hold the value of Sol_NN and its first n-1 derivatives at each
    # collocation point. The ith row of this Tensor holds the value of Sol_NN
    # and its first n-1 derivatives at the ith collocation point. Its jth column
    # holds the jth spatial derivative of Sol_NN at each collocation point.
    num_Collocation_Points : int = Coords.shape[0];
    diu_dxi_batch                = torch.empty((num_Collocation_Points, num_derivatives + 1),
                                                dtype = Data_Type,
                                                device = Device);

    # Calculate approximate solution at this collocation point.
    diu_dxi_batch[:, 0] = Sol_NN(Coords).squeeze();

    # Compute the derivative of Sol_NN with respect to t, x at each collocation
    # point. To speed up computations, we batch this computation. It's
    # important, however, to take a close look at what this is doing (because
    # it's not very obvious). Let N denote the number of collocation points.
    # Let's focus on how Torch computes the derivative of u with respect to x.
    # Let the Jacobian matrix J be defined as follows:
    #       | (d/dx_0)u(t_0, x_0),  (d/dx_0)u(t_1, x_1),... (d/dx_0)u(t_N, x_N) |
    #       | (d/dx_1)u(t_0, x_0),  (d/dx_1)u(t_1, x_1),... (d/dx_1)u(t_N, x_N) |
    #   J = |  ....                  ....                    ....               |
    #       | (d/dx_N)u(t_0, x_0),  (d/dx_N)u(t_1, x_1),... (d/dx_N)u(t_N, x_N) |
    # Let's focus on the jth column. Here, we compute the derivative of
    # u(t_j, x_j) with respect to x_0, x_1,.... x_N. Since u(t_j, x_j) only
    # depends on x_j (because of how its computational graph was constructed),
    # all of these derivatives will be zero except for the jth one. More
    # broadly, this means that J will be a diagonal matrix. When we compute
    # torch.autograd.grad with a non-scalar outputs variable, we need to pass a
    # grad_outputs Tensor which has the same shape. Let v denote the vector we
    # pass as grad outputs. In our case, v is a vector of ones. Torch then
    # computes Jv. Since J is diagonal (by the argument above), the ith
    # component of this product is (d/dx_i)u(t_i, x_i), precisely what we want.
    # Torch does the same thing for derivatives with respect to t. The end
    # result is a 2 column Tensor. whose (i, 0) entry holds (d/dt_i)u(t_i, x_i),
    # and whose (i, 1) entry holds (d/dx_i)u(t_i, x_i).
    grad_u = torch.autograd.grad(
                outputs         = diu_dxi_batch[:, 0],
                inputs          = Coords,
                grad_outputs    = torch.ones_like(diu_dxi_batch[:, 0]),
                retain_graph    = True,
                create_graph    = True)[0];
    # So... why do we do this rather than computing the derivatives
    # point-by-point? Performance! This batched approach is much faster because
    # it takes advantage of memory locality.

    # extract du/dx and du/dt (at each collocation point) from grad_u.
    du_dt_batch         = grad_u[:, 0];
    diu_dxi_batch[:, 1] = grad_u[:, 1];

    # Compute higher order derivatives
    for i in range(2, num_derivatives + 1):
        # At each collocation point, compute d^(i-1)u(x, t/dx^(i-1) with respect
        # to t, x. This uses the same process as is described above for grad_u,
        # but with (d^(i-1)/dx^(i-1))u in place of u.
        # We need to create graphs for this so that torch can track this
        # operation when constructing the computational graph for the loss
        # function (which it will use in backpropagation). We also need to
        # retain grad_u's graph for backpropagation.
        grad_diu_dxi = torch.autograd.grad(
                        outputs         = diu_dxi_batch[:, i - 1],
                        inputs          = Coords,
                        grad_outputs    = torch.ones_like(diu_dxi_batch[:, i - 1]),
                        retain_graph    = True,
                        create_graph    = True)[0];

        # Extract (d^i/dx^i)u, which is the 1 column of the above Tensor.
        diu_dxi_batch[:, i] = grad_diu_dxi[:, 1];

    return (du_dt_batch, diu_dxi_batch);



def PDE_Residual(
        Sol_NN      : Neural_Network,
        PDE_NN      : Neural_Network,
        Coords    : torch.Tensor,
        Data_Type : torch.dtype = torch.float32,
        Device    : torch.device = torch.device('cpu')) -> torch.Tensor:
    """ This function evaluates the "PDE residual" at each coordinate in Coords.
    For brevtiy, let u = Sol_NN, and N = PDE_NN. At each coordinate, we compute
            du/dt - N(u, du/dx, d^2u/dx^2,... )
    which we call the residual.

    Note: this function only works if Sol_NN is a function of 1 spatial variable.

    ----------------------------------------------------------------------------
    Arguments:

    Sol_NN: The network that approximates the PDE solution.

    PDE_NN: The network that approximates the PDE.

    Coords: A two-column Tensor whose ith row holds the t, x coordinates of the
    ith point where we evaluate the PDE residual.

    Data_Type: The data type that all tensors in Sol_NN and PDE_NN use.

    Device: The device that Sol_NN and PDE_NN are loaded on.

    ----------------------------------------------------------------------------
    Returns:

    An M element Tensor (where M is the number of Coords) whose ith entry holds
    the residual at the ith coordinate. """


    # Determine how many derivatives of Sol_NN we'll need to evaluate the PDE.
    # Remember that PDE_NN is a function of u, du/dx, d^2u/dx^2,
    # d^(n-1)u/dx^(n-1), where n is the number of inputs that PDE_NN accepts.
    # For brevity, let u = Sol_NN.  Thus, the number of derivatives is the number
    # of inputs for PDE_NN minus  1.  Once we know this, we evaluate du/dt, u,
    # and the first n-1 spatial derivatives of u at each collocation point.
    num_derivatives             = PDE_NN.Input_Dim - 1;
    du_dt_batch, diu_dxi_batch  = Evaluate_Sol_Derivatives(
                                        Sol_NN          = Sol_NN,
                                        num_derivatives = num_derivatives,
                                        Coords          = Coords,
                                        Data_Type       = Data_Type,
                                        Device          = Device);

    # Evaluate PDE_NN at each row of diu_dxi. This yields an N by 1 Tensor
    # (where N is the number of rows in Coords) whose ith row holds the value of
    # N at (u(t_i, x_i), (d/dx)u(t_i, x_i),... (d^(n-1))/dx^(n-1))u(t_i, x_i)).
    # We squeeze this to eliminate the extra dimension.
    PDE_NN_batch = PDE_NN(diu_dxi_batch).squeeze();

    # At each Collocation point, evaluate the square of the residuals
    # du/dt - N(u, du/dx,... ).
    return (du_dt_batch - PDE_NN_batch);
