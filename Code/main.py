import torch;
import numpy as np;
from typing import Tuple;
import matplotlib.pyplot as plt;

from Network import Neural_Network, Collocation_Loss, Data_Loss;
from Plotter import Update_Axes, Setup_Axes;
from Setup_File_Reader import Setup_File_Reader, Setup_Data_Container;
from Data_Loader import Data_Loader, Data_Container;



def Discovery_Training(
        u_NN                : Neural_Network,
        N_NN                : Neural_Network,
        Collocation_Coords  : torch.Tensor,
        Data_Coords         : torch.Tensor,
        Data_Values         : torch.Tensor,
        Optimizer           : torch.optim.Optimizer) -> None:
    """ This function runs one epoch of training when in "Discovery" mode. In
    this mode, we enforce the leaned PDE at the Collocation_Points, and the
    Data_Values at the Data_Points.

    ----------------------------------------------------------------------------
    Arguments:
    u_NN : Neural network that approximates the solution to the learned PDE.

    N_NN : Neural network that approximates the PDE.

    Collocation_Coords : the collocation points at which we enforce the learned
    PDE. This should be a 2 column tensor of floats whose ith holds the x,t
    coordinates of the ith collocation point.

    Data_Coords : A tensor holding the coordinates of the points at which we
    compare the approximate solution to the true one. This should be a 2 column
    tensor whose ith row holds the x,t coordinates of the ith data point.

    Data_Values : A tensor holding the value of the true solution at the data
    points. If Data_Coords has N rows, then this should be an N element tensor
    of floats whose ith element holds the value of the true solution at the ith
    data point.

    optimizer : the optimizer we use to train u_NN and N_NN. It should be
    loaded with the gradients of both networks.

    ----------------------------------------------------------------------------
    returns:
    Nothing! """

    # Zero out the gradients.
    Optimizer.zero_grad();

    # Evaluate the Loss (Note, we enforce a BC of 0)
    Loss = (Collocation_Loss(u_NN = u_NN, N_NN = N_NN, Collocation_Coords = Collocation_Coords) +
            Data_Loss(u_NN = u_NN, Data_Coords = Data_Coords, Data_Values = Data_Values));

    # Back-propigate to compute gradients of Loss with respect to network
    # parameters.
    Loss.backward();

    # update network weights.
    Optimizer.step();



def Discovery_Testing(
        u_NN                : Neural_Network,
        N_NN                :  Neural_Network,
        Collocation_Coords  : torch.Tensor,
        Data_Coords         : torch.Tensor,
        Data_Values         : torch.Tensor) -> Tuple[float, float]:
    """ This function runs testing when in "Discovery" mode. You CAN NOT run this
    function with no_grad set True. Why? Because we need to evaluate derivatives
    of the solution with respect to the inputs! Thus, we need torch to build a
    cmputational graph.

    ----------------------------------------------------------------------------
    Arguments:
    u_NN : Neural network that approximates the solution to the learned PDE.

    N_NN : Neural network that approximates the learned PDE.

    Collocation_Coords : the collocation points at which we enforce the learned
    PDE. This should be a 2 column tensor of floats whose ith holds the x,t
    coordinates of the ith collocation point.

    Data_Coords : A tensor holding the coordinates of the points at which we
    compare the approximate solution to the true one. This should be a 2 column
    tensor whose ith row holds the x,t coordinates of the ith data point.

    Data_Values : A tensor holding the value of the true solution at the data
    points. If Data_Coords has N rows, then this should be an N element tensor
    of floats whose ith element holds the value of the true solution at the ith
    data point.

    ----------------------------------------------------------------------------
    Returns:
    a tuple of floats. The first element holds the collocation loss, while
    the second holds the data loss. """

    # Get the losses at the passed collocation points (Note we enforce a 0 BC)
    Coloc_Loss : float = Collocation_Loss(
                            u_NN = u_NN,
                            N_NN = N_NN,
                            Collocation_Coords = Collocation_Coords).item();
    Data_loss : float  = Data_Loss(
                            u_NN = u_NN,
                            Data_Coords = Data_Coords,
                            Data_Values = Data_Values).item();

    # Return the losses.
    return (Coloc_Loss, Data_loss);

    # Should we worry about the computational graph that we build in this
    # function? No. Here's why:
    # Cmputing the losses requires propigating the inputs through the network,
    # thereby building up a computational graph (we need to keep the graph
    # building enabled b/c we have to evaluate derivatives to compute the
    # collocation loss). Normally, these graphs are freed when we call backward.
    # That's what happens in the training loop. Here, we don't use backward.
    # The graphs will be freed, however. This function builds up graphs for
    # Coloc_Loss and Data_loss. When this function returns, however, both
    # variables are freed (along with their graphs!). These graphs do not get
    # passed Coloc_Loss or Data_loss, since both are floats (not Tensors).



def PINNs_Training(
        u_NN                        : Neural_Network,
        N_NN                        : Neural_Network,
        IC_Coords                   : torch.Tensor,
        IC_Data                     : torch.Tensor,
        Lower_Bound_Coords          : torch.Tensor,
        Upper_Bound_Coords          : torch.Tensor,
        Periodic_BCs_Highest_Order  : int,
        Collocation_Coords          : torch.Tensor,
        Data_Coords                 : torch.Tensor,
        Data_Values                 : torch.Tensor,
        Optimizer                   : torch.optim.Optimizer) -> None:
    """ This function runs one epoch of training when in "PINNs" mode. In
    this mode, we enforce the leaned PDE at the Collocation_Points, impose
    Initial Conditions (ICs), and Periodic Boundary Condtions (BCs).

    ----------------------------------------------------------------------------
    Arguments:
    u_NN : Neural network that approximates the solution to the learned PDE.

    N_NN : Neural network that approximates the PDE.

    IC_Coords : A tensor that holds the coordinates of each point that we
    enforce the Initial Condition. This should be a 2 column tensor of floats
    whose ith row holds the x,t coordinates of the ith point where we enforce
    the IC.

    IC_Data : A tensor that holds the value of the initial condition at each
    point in IC_Coords. If IC_Coords has N rows, then this should be an N
    element tensor whose ith entry holds the value of the IC at the ith IC
    point.

    Lower_Bound_Coords : A tensor that holds the coordinates of each grid point
    on the lower spatial bound of the domain.

    Uppder_Bound_Coords : A tensor that holds the coordinates of each grid point
    on the lower spatial bound of the domain.

    Periodic_BCs_Highest_Order : If this is set to N, then we will enforce
    periodic BCs on the solution and its first N-1 derivatives.

    Collocation_Coords : the collocation points at which we enforce the learned
    PDE. This should be a 2 column tensor of floats whose ith holds the x,t
    coordinates of the ith collocation point.

    Data_Coords : A tensor holding the coordinates of the points at which we
    compare the approximate solution to the true one. This should be a 2 column
    tensor whose ith row holds the x,t coordinates of the ith data point.

    Data_Values : A tensor holding the value of the true solution at the data
    points. If Data_Coords has N rows, then this should be an N element tensor
    of floats whose ith element holds the value of the true solution at the ith
    data point.

    optimizer : the optimizer we use to train u_NN.

    ----------------------------------------------------------------------------
    returns:
    Nothing! """

    # Zero out the gradients.
    Optimizer.zero_grad();

    # Evaluate the Loss (Note, we enforce a BC of 0)
    Loss = (IC_Loss(
                u_NN = u_NN,
                N_NN = N_NN,
                IC_Coords = IC_Coords,
                IC_Data = IC_Data)

            +

            Periodic_BC_Loss(
                u_NN = u_NN,
                N_NN = N_NN,
                Lower_Bound_Coords = Lower_Bound_Coords,
                Upper_Bound_Coords = Upper_Bound_Coords,
                Highest_Order = Periodic_BCs_Highest_Order)
            +

            Collocation_Loss(
                u_NN = u_NN,
                N_NN = N_NN,
                Collocation_Coords = Collocation_Coords));

    # Back-propigate to compute gradients of Loss with respect to network
    # parameters
    Loss.backward();

    # update network weights.
    Optimizer.step();



def PINNs_Training(
        u_NN                        : Neural_Network,
        N_NN                        : Neural_Network,
        IC_Coords                   : torch.Tensor,
        IC_Data                     : torch.Tensor,
        Lower_Bound_Coords          : torch.Tensor,
        Upper_Bound_Coords          : torch.Tensor,
        Periodic_BCs_Highest_Order  : int,
        Collocation_Coords          : torch.Tensor,
        Data_Coords                 : torch.Tensor,
        Data_Values                 : torch.Tensor) -> Tuple[float, float, float]:
    """ This function runs one epoch of testing when in "PINNs" mode. In
    this mode, we enforce the leaned PDE at the Collocation_Points, impose
    Initial Conditions (ICs), and Periodic Boundary Condtions (BCs).

    ----------------------------------------------------------------------------
    Arguments:
    u_NN : Neural network that approximates the solution to the learned PDE.

    N_NN : Neural network that approximates the PDE.

    IC_Coords : A tensor that holds the coordinates of each point that we
    enforce the Initial Condition. This should be a 2 column tensor of floats
    whose ith row holds the x,t coordinates of the ith point where we enforce
    the IC.

    IC_Data : A tensor that holds the value of the initial condition at each
    point in IC_Coords. If IC_Coords has N rows, then this should be an N
    element tensor whose ith entry holds the value of the IC at the ith IC
    point.

    Lower_Bound_Coords : A tensor that holds the coordinates of each grid point
    on the lower spatial bound of the domain.

    Uppder_Bound_Coords : A tensor that holds the coordinates of each grid point
    on the lower spatial bound of the domain.

    Periodic_BCs_Highest_Order : If this is set to N, then we will enforce
    periodic BCs on the solution and its first N-1 derivatives.

    Collocation_Coords : the collocation points at which we enforce the learned
    PDE. This should be a 2 column tensor of floats whose ith holds the x,t
    coordinates of the ith collocation point.

    Data_Coords : A tensor holding the coordinates of the points at which we
    compare the approximate solution to the true one. This should be a 2 column
    tensor whose ith row holds the x,t coordinates of the ith data point.

    Data_Values : A tensor holding the value of the true solution at the data
    points. If Data_Coords has N rows, then this should be an N element tensor
    of floats whose ith element holds the value of the true solution at the ith
    data point.

    ----------------------------------------------------------------------------
    returns:
    A tuple of three floats. The 0 element holds the IC loss, the 1 element
    holds the BC loss, the 2 element holds the Collocation loss. """

    # Get the losses at the passed collocation points (Note we enforce a 0 BC)
    IC_Loss : float    = IC_Loss(
                            u_NN = u_NN,
                            N_NN = N_NN,
                            IC_Coords = IC_Coords,
                            IC_Data = IC_Data);

    BC_Loss : float    = Periodic_BC_Loss(
                            u_NN = u_NN,
                            N_NN = N_NN,
                            Lower_Bound_Coords = Lower_Bound_Coords,
                            Upper_Bound_Coords = Upper_Bound_Coords,
                            Highest_Order = Periodic_BCs_Highest_Order).item();

    Coloc_Loss : float = Collocation_Loss(
                            u_NN = u_NN,
                            N_NN = N_NN,
                            Collocation_Coords = Collocation_Coords).item();

    # Return the losses.
    return (IC_Loss, BC_Loss, Coloc_Loss);



def Setup_Optimizer(
        u_NN            : Neural_Network,
        N_NN            : Neural_Network,
        Mode            : str,
        Learning_Rate   : float) -> torch.optim.Optimizer:
    """ This function sets up the optimizer depending on if the N Network has
    learning enabled or not. It also disables gradients for all network
    parameters that are not being learned.

    ----------------------------------------------------------------------------
    Arguments :

    u_NN : The neural network that approximates the solution.

    N_NN : The neural network that approximates the PDE.

    Mode : Controls which mode the code is running in. Either "Discovery" or
    "PINNs". If "Discovery", then both u_NN and N_NN are learned. Otherwise,
    N_NN is fixed and u_NN is learned.

    Learning_Rate : the desired learning rate.

    ----------------------------------------------------------------------------
    Returns:
    The optimizer! """

    if(Mode == "Discovery"):
        # We need to train both u_NN and N_NN. Pass both networks' parameters to
        # the optimizer.
        return torch.optim.Adam(list(u_NN.parameters()) + list(N_NN.parameters()), lr = Learning_Rate);
    elif(Mode == "PINNs"):
        # If we're in PINNs mode, then N_NN does not require gradients.
        N_NN.requires_grad_(False);

        # Setup the optimizer using only u_NN's parameters.
        return torch.optim.Adam(u_NN.parameters(), lr = Learning_Rate);
    else:
        print("Mode is neither \"Discovery\" nor \"PINNs\". Something went wrong. Aborting");
        exit();



def main():
    # Load setup data from the setup file.
    Setup_Data = Setup_File_Reader();

    # Test that we got the correct input.
    print("Loaded the following settings:");
    for item in Setup_Data.__dict__.items():
        print(item);

    # Initialize Network hyperparameters.
    Epochs        : int   = Setup_Data.Epochs;
    Learning_Rate : float = Setup_Data.Learning_Rate;

    # Set up the neural network to approximate the PDE solution.
    u_NN = Neural_Network(  Num_Hidden_Layers   = Setup_Data.u_Num_Hidden_Layers,
                            Nodes_Per_Layer     = Setup_Data.u_Nodes_Per_Layer,
                            Input_Dim           = 2,
                            Output_Dim          = 1);

    # Set up the neural network to approximate the PDE operator N.
    N_NN = Neural_Network(  Num_Hidden_Layers   = Setup_Data.N_Num_Hidden_Layers,
                            Nodes_Per_Layer     = Setup_Data.N_Nodes_Per_Layer,
                            Input_Dim           = Setup_Data.N_Num_u_derivatives + 1,
                            Output_Dim          = 1);

    # Select the optimizer based on mode.
    Optimizer = Setup_Optimizer(u_NN = u_NN,
                                N_NN = N_NN,
                                Mode = Setup_Data.Mode,
                                Learning_Rate = Learning_Rate );

    # Check if we're loading anything from file.
    if(     Setup_Data.Load_u_Network_State == True or
            Setup_Data.Load_N_Network_State == True or
            Setup_Data.Load_Optimize_State  == True):

        # Load the saved checkpoint.
        Load_File_Path : str = "../Saves/" + Setup_Data.Load_File_Name;
        Saved_State = torch.load(Load_File_Path);

        if(Setup_Data.Load_u_Network_State == True):
            u_NN.load_state_dict(Saved_State["u_Network_State"]);
            u_NN.train();

        if(Setup_Data.Load_N_Network_State == True):
            N_NN.load_state_dict(Saved_State["N_Network_State"]);
            N_NN.train();

        # Note that this will overwrite the Learning Rate using the
        # Learning rate in the saved state. Thus, if this is set to true, then
        # we essentially ignore the learning rate in the setup file.
        if(Setup_Data.Load_Optimize_State  == True):
            Optimizer.load_state_dict(Saved_State["Optimizer_State"]);

    # Set up training and training collocation/boundary points.
    Data_Container = Data_Loader(Setup_Data.Mode, "../Data/" + Setup_Data.Data_File_Name, Setup_Data.Num_Training_Points, Setup_Data.Num_Testing_Points);

    # Loop through the epochs.
    if(Setup_Data.Mode == "Discovery"):
        # Set up array for the different kinds of losses.
        Collocation_Losses = np.empty((Epochs), dtype = np.float);
        Data_Losses        = np.empty((Epochs), dtype = np.float);

        for t in range(Epochs):
            Discovery_Training(
                u_NN                = u_NN,
                N_NN                = N_NN,
                Collocation_Coords  = Data_Container.Train_Coloc_Coords,
                Data_Coords         = Data_Container.Train_Data_Coords,
                Data_Values         = Data_Container.Train_Data_Values,
                Optimizer           = Optimizer);

            (Collocation_Losses[t], Data_Losses[t]) = Discovery_Testing(
                u_NN                = u_NN,
                N_NN                = N_NN,
                Collocation_Coords  = Data_Container.Test_Coloc_Coords,
                Data_Coords         = Data_Container.Test_Data_Coords,
                Data_Values         = Data_Container.Test_Data_Values );

            # Print losses.
            print(("Epoch #%-4d: "              % t)                    , end = '');
            print(("\tCollocation Loss = %7f"   % Collocation_Losses[t]), end = '');
            print((",\t Data Loss = %7f"        % Data_Losses[t])       , end = '');
            print((",\t Total Loss = %7f"       % (Collocation_Losses[t] + Data_Losses[t])));

    elif(Setup_Data.Mode == "PINNs"):
        # Setup arrays for the diffeent kinds of losses.
        IC_Losses          = np.empty((Epochs), dtype = np.float);
        BC_Losses          = np.empty((Epochs), dtype = np.float);
        Collocation_Losses = np.empty((Epochs), dtype = np.float);

        for t in range(Epochs):
            PINNs_Training(
                u_NN                        = u_NN,
                N_NN                        = N_NN,
                IC_Coords                   = Data_Container.IC_Coords,
                IC_Data                     = Data_Container.IC_Data,
                Lower_Bound_Coords          = Data_Container.Lower_Bound_Coords,
                Upper_Bound_Coords          = Data_Container.Upper_Bound_Coords,
                Periodic_BCs_Highest_Order  = Setup_Data.Periodic_BCs_Highest_Order,
                Collocation_Coords          = Data_Container.Train_Coloc_Coords,
                Data_Coords                 = Data_Container.Train_Data_Coords,
                Data_Values                 = Data_Container.Train_Data_Values,
                Optimizer                   = Optimizer);

            (IC_Losses[t], BC_Losses[t], Collocation_Losses[t]) = PINNs_Testing(
                u_NN                        = u_NN,
                N_NN                        = N_NN,
                IC_Coords                   = Data_Container.IC_Coords,
                IC_Data                     = Data_Container.IC_Data,
                Lower_Bound_Coords          = Data_Container.Lower_Bound_Coords,
                Upper_Bound_Coords          = Data_Container.Upper_Bound_Coords,
                Periodic_BCs_Highest_Order  = Setup_Data.Periodic_BCs_Highest_Order,
                Collocation_Coords          = Data_Container.Test_Coloc_Coords,
                Data_Coords                 = Data_Container.Test_Data_Coords,
                Data_Values                 = Data_Container.Test_Data_Values );

            # Print losses.
            print(("Epoch #%-4d: "              % t)                    , end = '');
            print(("\IC Loss = %7f"             % IC_Losses[t])         , end = '');
            print(("\BC Loss = %7f"             % BC_Losses[t])         , end = '');
            print(("\tCollocation Loss = %7f"   % Collocation_Losses[t]), end = '');
            print((",\t Total Loss = %7f"       % (IC_Losses[t] + BC_Losses[t] + Collocation_Losses[t])));

    # Save the network and optimizer states!
    if(Setup_Data.Save_To_File == True):
        Save_File_Path : str = "../Saves/" + Setup_Data.Save_File_Name;
        torch.save({"u_Network_State" : u_NN.state_dict(),
                    "N_Network_State" : N_NN.state_dict(),
                    "Optimizer_State" : Optimizer.state_dict()},
                    Save_File_Path);

    # Plot final results (if we should)
    if(Setup_Data.Plot_Final_Results == True):
        fig, Axes = Setup_Axes();
        Update_Axes(fig                 = fig,
                    Axes                = Axes,
                    u_NN                = u_NN,
                    N_NN                = N_NN,
                    x_points            = Data_Container.x_points,
                    t_points            = Data_Container.t_points,
                    True_Sol_On_Grid    = Data_Container.True_Sol_On_Grid);
        plt.show();


if __name__ == '__main__':
    main();
