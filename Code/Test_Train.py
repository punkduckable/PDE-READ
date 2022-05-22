import numpy as np;
import torch;
from typing import Tuple;

from Network import Neural_Network;
from Loss_Functions import Data_Loss, Collocation_Loss;



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
