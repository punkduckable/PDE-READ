import numpy as np;
import torch;

from typing import Tuple;
from Network import Neural_Network;



def Evaluate_u_derivatives(
        u_NN            : Neural_Network,
        num_derivatives : int,
        Coords          : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """ This function evaluates u, du/dt, and d^i u/dx^i (for i = 1,2... ) at each
    coordinate in Coords.

    ----------------------------------------------------------------------------
    Arguments:
    u_NN : The neural network that approximates the solution.

    num_derivative : The number of spatial derivatives of u_NN we need to
    evaluate. num_derivative - 1 is the highest order derivative we will
    evaluate.

    Coords : A M by 2 tensor of coordinates. The ith row of this tensor should
    contain the ith x, t coordinates.

    ----------------------------------------------------------------------------
    Returns:
    This returns a two element Tuple! If Coords is a M by 2 tensor, then thie
    first returned argument is a M element tensor whose ith element holds the
    value of du/dt at the ith coordinate. If N_NN accepts N argumetns, then
    the second return variable is a M by N tensor whose i,j element holds the
    value of d^ju/dx^j at the ith coordinate. """

    # We need to evaluate derivatives, so set Requires Grad to true.
    Coords.requires_grad_(True);

    # Determine how many derivatives of u we'll need to evaluate the PDE.
    # Remember that N is a function of u, du/dx, d^2u/dx^2, d^(n-1)u/dx^(n-1),
    # where n is the number of inputs that N_NN accepts. Once we know this,
    # and the Number of Collocation points, we initialize a tensor to hold the
    # value of u and its first n-1 derivatives at each collocation point.
    # The ith row of this tensor holds the value of u and its first n-1
    # derivatives at the ith collocation point. Its jth column holds the jth
    # spatial derivative of u at each collocation point.
    num_Collocation_Points : int = Coords.shape[0];
    diu_dxi_batch                = torch.empty((num_Collocation_Points, num_derivatives), dtype = torch.float32);

    # Calculate approximate solution at this collocation point.
    diu_dxi_batch[:, 0] = u_NN(Coords).squeeze();

    # Compute the derivative of the NN output with respect to x, t at each
    # collocation point. To speed up computations, we batch this computation.
    # It's important, however, to take a close look at what this is doing
    # (because it's not terribly obvious). Let N denote the number of
    # collocation points. Let's focus on how torch computes the derivative of
    # u with respect to x. Let the Jacobian matrix J be defined as follows:
    #       | (d/dx_0)u(x_0, t_0),  (d/dx_0)u(x_1, t_1),... (d/dx_0)u(x_N, t_N) |
    #       | (d/dx_1)u(x_0, t_0),  (d/dx_1)u(x_1, t_1),... (d/dx_1)u(x_N, t_N) |
    #   J = |  ....                  ....                    ....               |
    #       | (d/dx_N)u(x_0, t_0),  (d/dx_N)u(x_1, t_1),... (d/dx_N)u(x_N, t_N) |
    # Let's focus on the jth column. Here we compute the derivative of
    # u(x_j, t_j) with respect to x_0, x_1,.... x_N. Since u(x_j, t_j) only
    # depends on x_j (because of how its computational graph was constructed),
    # all of these derivatives will be zero except for the jth one. More
    # broadly, this means that J will be a diagonal matrix. When we compute
    # torch.autograd.grad with a non-scalar outputs variable, we need to pass a
    # grad_outputs tensor which has the same shape. Let v denote the vector we
    # pass as grad outputs. In our case, v is a vector of ones. Torch then
    # computes Jv. Since J is diagonal (by the argument above), the ith
    # component of this product is (d/dx_i)u(x_i, t_i), precisely what we want.
    # Torch does the same thing for derivatives with respect to t. The end
    # result is a 2 column tensor. The ith entry of the 0 column holds
    # (d/dx_i)u(x_i, t_i), while the ith entry of the 1 column holds
    # (d/dt_i)u(x_i, t_i).
    grad_u = torch.autograd.grad(
                outputs         = diu_dxi_batch[:, 0],
                inputs          = Coords,
                grad_outputs    = torch.ones_like(diu_dxi_batch[:, 0]),
                retain_graph    = True,
                create_graph    = True)[0];
    # So.... why do we do this rather than compute the
    # derivatives point-by-point? Performance! This batched approach is
    # much faster because it can take advantage of memory locality.

    # extract du/dx and du/dt (at each collocation point) from grad_u.
    diu_dxi_batch[:, 1] = grad_u[:, 0];
    du_dt_batch         = grad_u[:, 1];

    # Compute higher order derivatives
    for i in range(2, num_derivatives):
        # At each collocation point, compute d^(i-1)u(x, t/dx^(i-1) with respect
        # to x, t. This uses the same process as is described above for grad_u,
        # but with (d^(i-1)/dx^(i-1))u in place of u.
        # We need to create graphs for this so that torch can track this
        # operation when constructing the computational graph for the loss
        # function (which it will use in backpropigation). We also need to
        # retain grad_u's graph for when we do backpropigation.
        grad_diu_dxi = torch.autograd.grad(
                        outputs         = diu_dxi_batch[:, i - 1],
                        inputs          = Coords,
                        grad_outputs    = torch.ones_like(diu_dxi_batch[:, i - 1]),
                        retain_graph    = True,
                        create_graph    = True)[0];

        # Extract (d^i/dx^i)u, which is the 0 column of the above tensor.
        diu_dxi_batch[:, i] = grad_diu_dxi[:, 0];

    return (du_dt_batch, diu_dxi_batch);



def PDE_Residual(
        u_NN    : Neural_Network,
        N_NN    : Neural_Network,
        Coords  : torch.Tensor) -> torch.Tensor:
    """ This function evaluates the "residual" of the PDE at a set of
    coordinates. For brevtiy, let u = u_NN, and N = N_NN. At each coordinate,
    this function computes
            du/dt - N(u, du/dx, d^2u/dx^2,... )
    which we call the residual.

    ----------------------------------------------------------------------------
    Arguments:
    u_NN : The neural network that approximates the solution.

    N_NN : The neural network that approximates the PDE.

    Coords : A M by 2 tensor of coordinates. The ith row of this tensor should
    contain the ith x, t coordinates.

    ----------------------------------------------------------------------------
    Returns:
    A M element tensor (where M is the number of collocation points) whose ith
    entry holds the residual at the ith coordinate.  """


    # Determine how many derivatives of u we'll need to evaluate the PDE.
    # Remember that N is a function of u, du/dx, d^2u/dx^2, d^(n-1)u/dx^(n-1),
    # where n is the number of inputs that N_NN accepts. Once we know this,
    # evaluate du/dt, u, and the first n-1 spatial derivatives of u at each
    # collocation point.
    num_derivatives             = N_NN.Input_Dim;
    du_dt_batch, diu_dxi_batch  = Evaluate_u_derivatives(
                                    u_NN            = u_NN,
                                    num_derivatives = num_derivatives,
                                    Coords          = Coords);

    # Evaluate N at each collocation point (ith row of diu_dxi). This results
    # a N by 1 tensor (where N is the number of collocation points) whose ith
    # row holds the value of N at (u(x_i, t_i), (d/dx)u(x_i, t_i),...
    # (d^(n-1))/dx^(n-1))u(x_i, t_i)). We squeeze it to get rid of the extra
    # (useless) dimension.
    N_NN_batch = N_NN(diu_dxi_batch).squeeze();

    # At each Collocation point, evaluate the square of the residuals
    # du/dt - N(u, du/dx,... ).
    return (du_dt_batch - N_NN_batch);
