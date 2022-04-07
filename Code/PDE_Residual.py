import numpy as np;
import torch;

from typing import Tuple;
from Network import Neural_Network;



def Evaluate_Derivatives(
        Sol_NN                      : Neural_Network,
        Time_Derivative_Order       : int,
        Spatial_Derivative_Order    : int,
        Coords                      : torch.Tensor,
        Data_Type                   : torch.dtype = torch.float32,
        Device                      : torch.device = torch.device('cpu')) -> Tuple[torch.Tensor, torch.Tensor]:
    """ This function evaluates U, D_t^m U (where m = Time_Derivative_Order)
    and D_x^i U (for i = 1,2,..., n. Where n = Spatial_Derivative_Order) at each
    coordinate in Coords.

    Note: This function only works if Sol_NN is a function of 1 spatial variable.

    ----------------------------------------------------------------------------
    Arguments:

    Sol_NN: The network that approximates the PDE solution.

    Time_Derivative_Order: The order of the time derivative on the left-hand
    side of the PDE.

    Spatial_Derivative_Order: The highest order spatial derivatives of Sol_NN we
    need to evaluate.

    Coords: A two-column Tensor whose ith row holds the t, x coordinates of the
    ith point we'll evaluate Sol_NN and its derivatives at.

    Data_Type: The data type that all Tensors in Sol_NN use.

    Device: The device that Sol_NN is loaded on.

    ----------------------------------------------------------------------------
    Returns:

    This returns a two-element Tuple! For brevity in what follows, let
    u = Sol_NN. If Coords is an M by 2 Tensor, then the first return argument
    is an M element Tensor whose ith element holds the value of D_t^m U at the
    ith coordinate. If PDE_NN accepts N arguments, then the second return
    variable is an M by N Tensor whose i,j element holds the value of D_x^j U
    at the ith coordinate. """

    # We need to evaluate derivatives, so set Requires Grad to true.
    Coords.requires_grad_(True);

    # Determine how many derivatives of Sol_NN we'll need to evaluate the PDE.
    # Remember that PDE_NN is a function of u, D_x U, D_x^2 U, ... D_x^{n}U,
    # where n = Spatial_Derivative_Order.
    # Once we know this, and the number of Collocation points, we initialize a
    # Tensor to hold the value of Sol_NN and its first n-1 derivatives at each
    # collocation point. The ith row of this Tensor holds the value of Sol_NN
    # and its first n-1 derivatives at the ith collocation point. Its jth column
    # holds the jth spatial derivative of Sol_NN at each collocation point.
    num_Collocation_Points : int = Coords.shape[0];
    Dxn_U                        = torch.empty((num_Collocation_Points, Spatial_Derivative_Order + 1),
                                                dtype  = Data_Type,
                                                device = Device);

    # Calculate approximate solution at this collocation point.
    Dxn_U[:, 0]                  = Sol_NN(Coords).view(-1);

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
    Grad_U = torch.autograd.grad(
                outputs         = Dxn_U[:, 0],
                inputs          = Coords,
                grad_outputs    = torch.ones_like(Dxn_U[:, 0]),
                retain_graph    = True,
                create_graph    = True)[0];
    # So... why do we do this rather than computing the derivatives
    # point-by-point? Performance! This batched approach is much faster because
    # it takes advantage of memory locality.

    # extract du/dx and du/dt (at each collocation point) from Grad_U.
    #Dtm_U : torch.Tensor    = Grad_U[:, 0].view(-1);
    Dxn_U[:, 1]             = Grad_U[:, 0];


    """
    # Compute the requested time derivative of U.
    for j in range(2, Time_Derivative_Order + 1):
        # At each coordinate, differentiate D_{t}^{i - 1} U with respect to
        # to t, x. This uses the same process we used for Grad_U (described
        # above), but with D_{t}^{i - 1} U in place of U. We need to create
        # graphs for this so that torch can track this operation when
        # constructing the computational graph for the loss function (which
        # it will use in backpropagation). We also need to retain Grad_U's
        # graph for back-propagation.
        Grad_Dtm_U = torch.autograd.grad(
                        outputs         = Dtm_U,
                        inputs          = Coords,
                        grad_outputs    = torch.ones_like(Dtm_U),
                        retain_graph    = True,
                        create_graph    = True)[0];

        # The 0 column should contain the ith time derivative of U.
        Dtm_U = Grad_Dtm_U[:].view(-1);"""


    # Compute higher order spatial derivatives
    for j in range(2, Spatial_Derivative_Order + 1):
        # At each collocation point, compute D_x^{j} - 1} U with respect
        # to t, x. This uses the same process as is described above for grad_u,
        # but with (D_x^{j - 1} U) in place of u.
        # We need to create graphs for this so that torch can track this
        # operation when constructing the computational graph for the loss
        # function (which it will use in backpropagation). We also need to
        # retain grad_u's graph for backpropagation.
        Grad_Dxn_U = torch.autograd.grad(
                        outputs         = Dxn_U[:, j - 1],
                        inputs          = Coords,
                        grad_outputs    = torch.ones_like(Dxn_U[:, j - 1]),
                        retain_graph    = True,
                        create_graph    = True)[0];

        # Extract D_x^i U, which is the 1 column of the above Tensor.
        Dxn_U[:, j] = Grad_Dxn_U[:, 0];

    return (Dxn_U);



def PDE_Residual(
        Sol_NN                      : Neural_Network,
        PDE_NN                      : Neural_Network,
        Time_Derivative_Order       : int,
        Spatial_Derivative_Order    : int,
        Coords                      : torch.Tensor,
        Data_Type                   : torch.dtype = torch.float32,
        Device                      : torch.device = torch.device('cpu')) -> torch.Tensor:
    """ This function evaluates the "PDE residual" at each coordinate in Coords.
    For brevtiy, let u = Sol_NN, and N = PDE_NN. At each coordinate, we compute
                        D_t^m U - N(u, D_x U, D_x^2 U, ... D_x^n U)
    which we call the residual. Here, m = Time_Derivative_Order and n =
    Spatial_Derivative_Order.

    Note: this function only works if Sol_NN is a function of 1 spatial variable.

    ----------------------------------------------------------------------------
    Arguments:

    Sol_NN: The network that approximates the PDE solution.

    PDE_NN: The network that approximates the PDE.

    Time_Derivative_Order: The order of the time derivative on the left-hand
    side of the PDE.

    Spatial_Derivative_Order: The highest order spatial derivative in the PDE.

    Coords: A two-column Tensor whose ith row holds the t, x coordinates of the
    ith point where we evaluate the PDE residual.

    Data_Type: The data type that all tensors in Sol_NN and PDE_NN use.

    Device: The device that Sol_NN and PDE_NN are loaded on.

    ----------------------------------------------------------------------------
    Returns:

    An M element Tensor (where M is the number of Coords) whose ith entry holds
    the residual at the ith coordinate. """


    # Determine how many derivatives of Sol_NN we'll need to evaluate the PDE.
    # Remember that PDE_NN is a function of u, D_x U, D_x^2 U, ... D_x^{n} U,
    # where n is one less than the input dimension of PDE_NN.
    # For brevity, let u = Sol_NN.  Thus, the number of derivatives is the number
    # of inputs for PDE_NN minus  1.  Once we know this, we evaluate du/dt, u,
    # and the first n-1 spatial derivatives of u at each collocation point.
    Dxn_U = Evaluate_Derivatives(
                        Sol_NN                      = Sol_NN,
                        Time_Derivative_Order       = Time_Derivative_Order,
                        Spatial_Derivative_Order    = Spatial_Derivative_Order,
                        Coords                      = Coords,
                        Data_Type                   = Data_Type,
                        Device                      = Device);

    # Evaluate PDE_NN at each row of diu_dxi. This yields an N by 1 Tensor
    # (where N is the number of rows in Coords) whose ith row holds the value of
    # N at (U(t_i, x_i), D_x U(t_i, x_i),... D_x^n U(t_i, x_i)).
    # We squeeze this to eliminate the extra dimension.
    PDE_NN_batch = PDE_NN(Dxn_U).view(-1);

    # At each Collocation point, evaluate the square of the residuals
    #       T - N(u, D_x U, ... , D_x^n U).
    # Here, T is the internal torque. Assuming a torque of +100 NM at 0, and
    # -10 NM at 20, this should be given by
    #       T(x) = -100 + (9/2)x
    T : torch.Tensor = -100.0 + (9./2.)*Coords.view(-1);

    return (T - PDE_NN_batch);
