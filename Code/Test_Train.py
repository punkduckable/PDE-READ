import numpy as np;
import torch;
from typing import Tuple;

from Network import Neural_Network;
from Loss_Functions import IC_Loss, Periodic_BC_Loss, Data_Loss, Collocation_Loss;



def Discovery_Training(
        Sol_NN                      : Neural_Network,
        PDE_NN                      : Neural_Network,
        Time_Derivative_Order       : int,
        Spatial_Derivative_Order    : int,
        Collocation_Coords          : torch.Tensor,
        Data_Coords                 : torch.Tensor,
        Data_Values                 : torch.Tensor,
        Optimizer                   : torch.optim.Optimizer,
        Data_Type                   : torch.dtype = torch.float32,
        Device                      : torch.device = torch.device('cpu')) -> None:
    """ This function runs one epoch of training when in "Discovery" mode. In
    this mode, we enforce the leaned PDE at the Collocation_Points and the
    Data_Values at the Data_Points.

    Note: This function works regardless of how many spatial variables Sol_NN
    depends on so long as Collocation_Loss does too.

    ----------------------------------------------------------------------------
    Arguments:

    Sol_NN: The network that approximates the PDE solution.

    PDE_NN: The network that approximates the PDE.

    Time_Derivative_Order: The order of the time derivative in the PDE we're
    trying to solve.

    Collocation_Coords: This should be a 2 column Tensor whose ith row holds the
    t, x coordinates of the ith collocation point.

    Collocation_Coords: the collocation points at which we enforce the learned
    PDE. If u accepts d spatial coordinates, then this should be a d+1 column
    tensor whose ith row holds the t, x_1,... x_d coordinates of the ith
    Collocation point.

    Data_Coords: A tensor holding the coordinates of the points at which we
    compare the approximate solution to the true one. If u accepts d spatial
    coordinates, then this should be a d+1 column tensor whose ith row holds the
    t, x_1,... x_d coordinates of the ith Datapoint.

    Data_Values: A tensor holding the value of the true solution at the data
    points. If Data_Coords has N rows, then this should be an N element tensor
    of floats whose ith element holds the value of the true solution at the ith
    data point.

    optimizer: the optimizer we use to train Sol_NN and PDE_NN. It should have
    been initialized with both network's parameters.

    Data_Type: The data type of all tensors in Sol_NN, PDE_NN.

    Device: The device for Sol_NN and PDE_NN.

    ----------------------------------------------------------------------------
    Returns:

    Nothing! """

    # Put the networks in training mode.
    Sol_NN.train();
    PDE_NN.train();

    # Define closure function (needed for LBFGS)
    def Discovery_Closure():
        # Zero out the gradients (if they are enabled).
        if (torch.is_grad_enabled()):
            Optimizer.zero_grad();

        # Evaluate the Loss (Note, we enforce a BC of 0)
        Loss = (Collocation_Loss(
                    Sol_NN                      = Sol_NN,
                    PDE_NN                      = PDE_NN,
                    Time_Derivative_Order       = Time_Derivative_Order,
                    Spatial_Derivative_Order    = Spatial_Derivative_Order,
                    Collocation_Coords          = Collocation_Coords,
                    Data_Type                   = Data_Type,
                    Device                      = Device)

                +

                Data_Loss(
                    Sol_NN = Sol_NN,
                    Data_Coords = Data_Coords,
                    Data_Values = Data_Values,
                    Data_Type = Data_Type,
                    Device    = Device));

        # Back-propigate to compute gradients of Loss with respect to network
        # parameters (only do if this if the loss requires grad)
        if (Loss.requires_grad == True):
            Loss.backward();

        return Loss;

    # update network parameters.
    Optimizer.step(Discovery_Closure);



def Discovery_Testing(
        Sol_NN                      : Neural_Network,
        PDE_NN                      : Neural_Network,
        Time_Derivative_Order       : int,
        Spatial_Derivative_Order    : int,
        Collocation_Coords          : torch.Tensor,
        Data_Coords                 : torch.Tensor,
        Data_Values                 : torch.Tensor,
        Data_Type                   : torch.dtype = torch.float32,
        Device                      : torch.device = torch.device('cpu')) -> Tuple[float, float]:
    """ This function runs testing when in "Discovery" mode. You CAN NOT run this
    function with no_grad set True. Why? Because we need to evaluate derivatives
    of the solution with respect to the inputs! Thus, we need torch to build a
    computational graph.

    Note: This function works regardless of how many spatial variables Sol_NN
    depends on so long as Collocation_Loss does too.

    ----------------------------------------------------------------------------
    Arguments:

    Sol_NN: The network that approximates the PDE solution.

    PDE_NN: The network that approximates the PDE.

    Time_Derivative_Order: The order of the time derivative in the PDE we're
    trying to solve.

    Collocation_Coords: This should be a 2 column Tensor whose ith row holds the
    t, x coordinates of the ith collocation point.

    Collocation_Coords: the collocation points at which we enforce the learned
    PDE. If u accepts d spatial coordinates, then this should be a d+1 column
    tensor whose ith row holds the t, x_1,... x_d coordinates of the ith
    Collocation point.

    Data_Coords: A tensor holding the coordinates of the points at which we
    compare the approximate solution to the true one. If u accepts d spatial
    coordinates, then this should be a d+1 column tensor whose ith row holds the
    t, x_1,... x_d coordinates of the ith Datapoint.

    Data_Values: A tensor holding the value of the true solution at the data
    points. If Data_Coords has N rows, then this should be an N element tensor
    of floats whose ith element holds the value of the true solution at the ith
    data point.

    Data_Type: The data type of all tensors in Sol_NN, PDE_NN.

    Device: The device for Sol_NN and PDE_NN.

    ----------------------------------------------------------------------------
    Returns:

    a tuple of floats. The first element holds the collocation loss, while
    the second holds the data loss. """

    # Put the networks in evaluation mode
    Sol_NN.eval();
    PDE_NN.eval();

    # Get the losses at the passed collocation points (Note we enforce a 0 BC)
    Coll_Loss : float = Collocation_Loss(
                            Sol_NN                      = Sol_NN,
                            PDE_NN                      = PDE_NN,
                            Time_Derivative_Order       = Time_Derivative_Order,
                            Spatial_Derivative_Order    = Spatial_Derivative_Order,
                            Collocation_Coords          = Collocation_Coords,
                            Data_Type                   = Data_Type,
                            Device                      = Device).item();

    Data_loss : float  = Data_Loss(
                            Sol_NN      = Sol_NN,
                            Data_Coords = Data_Coords,
                            Data_Values = Data_Values,
                            Data_Type   = Data_Type,
                            Device      = Device).item();

    # Return the losses.
    return (Coll_Loss, Data_loss);



def PINNs_Training(
        Sol_NN                      : Neural_Network,
        PDE_NN                      : Neural_Network,
        Time_Derivative_Order       : int,
        Spatial_Derivative_Order    : int,
        IC_Coords                   : torch.Tensor,
        IC_Data                     : torch.Tensor,
        Lower_Bound_Coords          : torch.Tensor,
        Upper_Bound_Coords          : torch.Tensor,
        Periodic_BCs_Highest_Order  : int,
        Collocation_Coords          : torch.Tensor,
        Optimizer                   : torch.optim.Optimizer,
        Data_Type                   : torch.dtype = torch.float32,
        Device                      : torch.device = torch.device('cpu')) -> None:
    """ This function runs one epoch of training when in "PINNs" mode. In this
    mode, we enforce the leaned PDE at the Collocation_Points, the Initial
    Conditions (ICs), and the Periodic Boundary Conditions (BCs).

    Note: This function works regardless of how many spatial variables Sol_NN
    depends on so long as Collocation_Loss and Periodic_BC_Loss do too.

    ----------------------------------------------------------------------------
    Arguments:

    Sol_NN: The network that approximates the PDE solution.

    PDE_NN: The network that approximates the PDE.

    Time_Derivative_Order: The order of the time derivative in the PDE we're
    trying to solve.

    Collocation_Coords: This should be a 2 column Tensor whose ith row holds the
    t, x coordinates of the ith collocation point.

    IC_Coords: A tensor that holds the coordinates of each point that we
    enforce the Initial Condition. If u accepts d spatial coordinates, then this
    should be a d+1 column tensor whose ith row holds the t, x_1,... x_d
    coordinates of the ith point where we'll enforce the IC.

    IC_Data: A tensor that holds the value of the initial condition at each
    point in IC_Coords. If IC_Coords has N rows, then this should be an N
    element tensor whose ith entry holds the value of the IC at the ith IC
    point.

    Lower_Bound_Coords: A tensor that holds the coordinates of each grid
    point on the lower spatial bound of the domain.

    Uppder_Bound_Coords: A tensor that holds the coordinates of each grid
    point on the lower spatial bound of the domain.

    Periodic_BCs_Highest_Order: If this is set to N, then we will enforce
    periodic BCs on the solution and its first N-1 derivatives.

    Collocation_Coords: the collocation points at which we enforce the learned
    PDE. If u accepts d spatial coordinates, then this should be a d+1 column
    tensor whose ith row holds the t, x_1,... x_d coordinates the ith
    Collocation point.

    Optimizer: the optimizer we use to train Sol_NN.

    Data_Type: The data type of all tensors in Sol_NN, PDE_NN.

    Device: The device for Sol_NN and PDE_NN.

    ----------------------------------------------------------------------------
    Returns:

    Nothing! """

    # Put the networks in training mode.
    Sol_NN.train();
    PDE_NN.train();

    # Define closure function (needed for LBFGS)
    def PINNs_Closure():
        # Zero out the gradients (if there are any).
        if (torch.is_grad_enabled()):
            Optimizer.zero_grad();

        # Evaluate the Loss (Note, we enforce a BC of 0)
        Loss = (IC_Loss(
                    Sol_NN    = Sol_NN,
                    IC_Coords = IC_Coords,
                    IC_Data   = IC_Data,
                    Data_Type = Data_Type,
                    Device    = Device)

                +

                Periodic_BC_Loss(
                    Sol_NN = Sol_NN,
                    Lower_Bound_Coords  = Lower_Bound_Coords,
                    Upper_Bound_Coords  = Upper_Bound_Coords,
                    Highest_Order       = Periodic_BCs_Highest_Order,
                    Data_Type           = Data_Type,
                    Device              = Device)

                +

                Collocation_Loss(
                    Sol_NN                      = Sol_NN,
                    PDE_NN                      = PDE_NN,
                    Time_Derivative_Order       = Time_Derivative_Order,
                    Spatial_Derivative_Order    = Spatial_Derivative_Order,
                    Collocation_Coords          = Collocation_Coords,
                    Data_Type                   = Data_Type,
                    Device                      = Device));

        # Back-propigate to compute gradients of Loss with respect to network
        # parameters (only do if this if the loss requires grad)
        if(Loss.requires_grad == True):
            Loss.backward();

    # update network parameters.
    Optimizer.step(PINNs_Closure);



def PINNs_Testing(
        Sol_NN                      : Neural_Network,
        PDE_NN                      : Neural_Network,
        Time_Derivative_Order       : int,
        Spatial_Derivative_Order    : int,
        IC_Coords                   : torch.Tensor,
        IC_Data                     : torch.Tensor,
        Lower_Bound_Coords          : torch.Tensor,
        Upper_Bound_Coords          : torch.Tensor,
        Periodic_BCs_Highest_Order  : int,
        Collocation_Coords          : torch.Tensor,
        Data_Type                   : torch.dtype = torch.float32,
        Device                      : torch.device = torch.device('cpu')) -> Tuple[float, float, float]:
    """ This function runs one epoch of testing when in "PINNs" mode. In this
    mode, we enforce the leaned PDE at the Collocation_Points, the Initial
    Conditions (ICs), and the Periodic Boundary Conditions (BCs).

    Note: This function works regardless of how many spatial variables Sol_NN
    depends on so long as Collocation_Loss and Periodic_BC_Loss do too.

    ----------------------------------------------------------------------------
    Arguments:

    Sol_NN: The network that approximates the PDE solution.

    PDE_NN: The network that approximates the PDE.

    Time_Derivative_Order: The order of the time derivative in the PDE we're
    trying to solve.

    Collocation_Coords: This should be a 2 column Tensor whose ith row holds the
    t, x coordinates of the ith collocation point.

    IC_Coords: A tensor that holds the coordinates of each point that we
    enforce the Initial Condition. If u accepts d spatial coordinates, then this
    should be a d+1 column tensor whose ith row holds the t, x_1,... x_d
    coordinates of the ith point where we'll enforce the IC.

    IC_Data: A tensor that holds the value of the initial condition at each
    point in IC_Coords. If IC_Coords has N rows, then this should be an N
    element tensor whose ith entry holds the value of the IC at the ith IC
    point.

    Lower_Bound_Coords: A tensor that holds the coordinates of each grid point
    on the lower spatial bound of the domain.

    Uppder_Bound_Coords: A tensor that holds the coordinates of each grid point
    on the lower spatial bound of the domain.

    Periodic_BCs_Highest_Order: If this is N, then we will enforce periodic BCs
    on the solution and its first N-1 derivatives.

    Collocation_Coords: the collocation points at which we enforce the learned
    PDE. If u accepts d spatial coordinates, then this should be a d+1 column
    tensor whose ith row holds the t, x_1,... x_d coordinates the ith
    Collocation point.

    Data_Type: The data type of all tensors in Sol_NN, PDE_NN.

    Device: The device for Sol_NN and PDE_NN.

    ----------------------------------------------------------------------------
    Returns:

    A tuple of three floats. The 0 element holds the Initial Condition loss,
    the 1 element holds the Boundary Condition loss, the 2 element holds the
    Collocation loss. """

    # Put the networks in evaluation mode
    Sol_NN.eval();
    PDE_Nn.eval();

    # Get the losses at the passed collocation points (Note we enforce a 0 BC)
    IC_Loss_Var : float     = IC_Loss(
                                Sol_NN    = Sol_NN,
                                IC_Coords = IC_Coords,
                                IC_Data   = IC_Data,
                                Data_Type = Data_Type,
                                Device    = Device).item();

    BC_Loss_Var : float     = Periodic_BC_Loss(
                                Sol_NN              = Sol_NN,
                                Lower_Bound_Coords  = Lower_Bound_Coords,
                                Upper_Bound_Coords  = Upper_Bound_Coords,
                                Highest_Order       = Periodic_BCs_Highest_Order,
                                Data_Type           = Data_Type,
                                Device              = Device).item();

    Col_Loss_Var : float    = Collocation_Loss(
                                Sol_NN                      = Sol_NN,
                                PDE_NN                      = PDE_NN,
                                Time_Derivative_Order       = Time_Derivative_Order,
                                Spatial_Derivative_Order    = Spatial_Derivative_Order,
                                Collocation_Coords          = Collocation_Coords,
                                Data_Type                   = Data_Type,
                                Device                      = Device).item();

    # Return the losses.
    return (IC_Loss_Var, BC_Loss_Var, Col_Loss_Var);
