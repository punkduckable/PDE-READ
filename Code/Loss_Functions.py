import numpy as np;
import torch;

from Network import Neural_Network;



# Loss from the initial condition.
def IC_Loss(
        u_NN : Neural_Network,
        IC_Coords : torch.Tensor,
        IC_Data : torch.Tensor) -> torch.Tensor:
    """ This function evaluates how well u_NN satisifies the initial condition.
    Specifically, for each point in IC_Coords, we evaluate u_NN. We then
    calculate the square of the difference between this and the corresponding
    true solution in IC_Data. We return the mean of these squared differences.

    ----------------------------------------------------------------------------
    Arguments:
    u_NN : the Neural Network that approximates the solution.

    IC_Coords : The coordinates where we know the true initial condition. This
    should be a Nx2 tensor whose ith row holds the x, t coodinates of the ith
    point.

    IC_Data : The value of the initial condition at each point in IC_Coords.
    This should be an N element tensor. """

    Num_IC_Points : int = IC_Coords.shape[0];

    Loss = torch.tensor(0, dtype = torch.float32);
    for i in range(Num_IC_Points):
        # Evaluate the Neural Network at the ith point.
        xt = IC_Coords[i];
        u_approx = u_NN(xt)[0];

        # Evaluate the square difference between the true and approx solution.
        u_true = IC_Data[i];
        Loss += (u_true - u_approx)**2;

    # Divide the accumulated loss by the number of IC points to get the mean
    # square error.
    return (Loss / Num_IC_Points);



# Loss from imposing periodic BCs
def Periodic_BC_Loss(
        u_NN : Neural_Network,
        Lower_Bound_Coords : torch.Tensor,
        Upper_Bound_Coords : torch.Tensor,
        Highest_Order : int) -> torch.Tensor:
    """ I need a description!!!!

    ENABLE MULTIPLE DERIVATIVES!!!!!! """

    Num_BC_Points = Lower_Bound_Coords.shape[0];

    Loss = torch.tensor(0, dtype = torch.float32);
    for i in range(Num_BC_Points):
        # evaluate the NN at the upper and lower bounds at the ith time
        # coordinate. We enable gradients because we will need gradients of u
        # with respect to xt_low and xt_high.
        xt_low = Lower_Bound_Coords[i];
        xt_low.requires_grad_(True);
        u_low = u_NN(xt_low);

        xt_high = Upper_Bound_Coords[i];
        xt_high.requires_grad_(True);
        u_high = u_NN(xt_high);

        # Set up tensors to hold the various derivatives of u at this
        # boundary point. The 0 element (corresponding to the zeroth order
        # derivative) is just the value of the function.
        u_low_derivatives  = torch.empty((Highest_Order+1), dtype = torch.float32);
        u_low_derivatives[0] = u_low;

        u_high_derivatives = torch.empty((Highest_Order+1), dtype = torch.float32);
        u_high_derivatives[0] = u_high;

        # Cycle through the higher order derivatives. For each derivative,
        # we compute d^ju/dx^j at the two boundaries. To do this, we first
        # compute the gradient of d^ju/dx^j with respect to xt. The 0
        # element of this result should hold the d^ju/dx^j.
        for j in range(1, Highest_Order + 1):
            grad_dju_dxj_low = torch.autograd.grad(
                                        u_low_derivatives[j-1],
                                        xt_low,
                                        retain_graph = True,
                                        create_graph = True)[0];
            u_low_derivatives[j] = grad_dju_dxj_low[0];

            grad_dju_dxj_high = torch.autograd.grad(
                                        u_high_derivatives[j-1],
                                        xt_high,
                                        retain_graph = True,
                                        create_graph = True)[0];
            u_high_derivatives[j] = grad_dju_dxj_high[0];

        # Now, accumulate the Periodic BC Loss at this point. For each j, we add
        # the square of the difference beteen the jth derivative at xt_low and
        # xt_high (which we want to be the same). That is,
        #           |d^ju/dx^j(xt_low) - d^ju/dx^j(xt_high)|^2
        for j in range(0, Highest_Order+1):
            Loss += (u_high_derivatives[j] - u_low_derivatives[j])**2;

    # Divide the accumulated loss by the number of BC points to get the mean
    # square error.
    return (Loss / Num_BC_Points);



# Evaluates du/dt - N(u, du/dx, d^u/dx^2,... ) at a specific point.
def PDE_Residual(
        u_NN : Neural_Network,
        N_NN : Neural_Network,
        xt : torch.Tensor) -> torch.Tensor:
    """ This function evaluates the "residual" of the PDE at a given point.
    For brevtiy, let u = u_NN, and N = N_NN. This function computes
            du/dt - N(u, du/dx, d^2u/dx^2,... )
    (which we call the residual) at the specified coodinate (x, t).

    ----------------------------------------------------------------------------
    Arguments:
    u_NN : The neural network that approximates the solution.

    N_NN : The neural network that approximates the PDE.

    xt : A 2 element tensor containing coordinates. The first element should be
    the x coordinate and the second should be the t coodinate.

    ----------------------------------------------------------------------------
    Returns :
    A single element tensor containing the residual. """

    # We need to compute the gradeint of u with respect to the x,t
    # coordinates.
    xt.requires_grad_(True);

    # Determine how many derivatives of u we'll need to evaluate the PDE.
    # Remember that N is a function of u, du/dx, d^2u/dx^2, d^(n-1)u/dx^(n-1),
    # where n is the number of inputs that N_NN accepts.
    n = N_NN.Input_Dim;
    u_derivatives = torch.empty((n), dtype = torch.float32);

    # Calculate approximate solution at this collocation point.
    u_derivatives[0] = u_NN(xt)[0];

    # Compute gradient of u with respect to xt. We have to create the graph
    # used to compute grad_u so that we can evaluate second derivatives.
    # We also need to set retain_graph to True (which is implicitly set by
    # setting create_graph = True, though I keep it to make the code more
    # explicit) so that torch keeps the computational graph for u, which we
    # will need when we do backpropigation.
    grad_u = torch.autograd.grad(u_derivatives[0], xt, retain_graph = True, create_graph = True)[0];

    # compute du/dx and du/dt. grad_u is a two element tensor. It's 0 element
    # holds du/dx, and its 1 element holds du/dt.
    u_derivatives[1] = grad_u[0];
    du_dt = grad_u[1];

    # Compute higher order derivatives
    for i in range(2, n):
        # Compute the gradient of d^(i-1)u/dx^(i-1) with respect to xt. We
        # need to create graphs for this so that torch can track this operation
        # when constructing the computational graph for the loss function
        # (which it will use in backpropigation). We also need to retain
        # grad_u's graph for when we do backpropigation.
        # The 0 element of this gradient holds d^iu/dx^i.
        grad_diu_dxi = torch.autograd.grad(u_derivatives[i-1], xt, retain_graph = True, create_graph = True)[0];
        u_derivatives[i] = grad_diu_dxi[0];

    # Evaluate the learned operator N at this point, use it to compute
    # du/dt - N(u, du/dx,... ).
    return du_dt - N_NN(u_derivatives)[0];



# Loss from enforcing the PDE at the collocation points.
def Collocation_Loss(
        u_NN : Neural_Network,
        N_NN : Neural_Network,
        Collocation_Coords : torch.Tensor) -> torch.Tensor:
    """ This function evaluates how well u_NN satisifies the learned PDE at the
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

    num_Collocation_Points : int = Collocation_Coords.shape[0];

    # Now, initialize the loss and loop through the collocation points!
    Loss = torch.tensor(0, dtype = torch.float32);
    for i in range(num_Collocation_Points):
        # Get the coordinates of the ith collocation point.
        xt = Collocation_Coords[i];

        # Evaluate the Residual from the learned PDE at this point.
        Loss += PDE_Residual(u_NN, N_NN, xt) ** 2;

    # Divide the accmulated loss by the number of collocation points to get
    # the mean square collocation loss.
    return (Loss / num_Collocation_Points);



# Loss from the training data.
def Data_Loss(
        u_NN : Neural_Network,
        Data_Coords : torch.Tensor,
        Data_Values : torch.Tensor) -> torch.Tensor:
    """ This function evaluates how well the learned solution u satisifies the
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

    num_Data_Points : int = Data_Coords.shape[0];

    # Now, initialize the Loss and loop through the Boundary Points.
    Loss = torch.tensor(0, dtype = torch.float32);
    for i in range(num_Data_Points):
        xt = Data_Coords[i];

        # Compute learned solution at this Data point.
        u_approx = u_NN(xt)[0];

        # Get exact solution at this data point.
        u_true = Data_Values[i];

        # Aggregate square of difference between the required BC and the learned
        # solution at this data point.
        Loss += (u_approx - u_true)**2;

    # Divide the accmulated loss by the number of boundary points to get
    # the mean square boundary loss.
    return (Loss / num_Data_Points);
