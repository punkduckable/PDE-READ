import torch;
import numpy as np;
from typing import Tuple;
import matplotlib.pyplot as plt;

from Network import Neural_Network, Collocation_Loss, Data_Loss;
from Plotter import Update_Axes, Generate_Plot_Gridpoints, Setup_Axes;
from Setup_File_Reader import Setup_File_Reader, Setup_Data_Container;
from Data_Loader import Data_Loader;



def Training_Loop(  u_NN : Neural_Network,
                    N_NN : Neural_Network,
                    Collocation_Coords : torch.Tensor,
                    Data_Coords : torch.Tensor,
                    Data_Values : torch.Tensor,
                    Optimizer : torch.optim.Optimizer) -> None:
    """ This loop runs one epoch of training for the neural network. In
    particular, we enforce the leaned PDE at the Collocation_Points, and the
    Data_Values at the Data_Points.

    ----------------------------------------------------------------------------
    Arguments:
    u_NN : Neural network that approximates the solution to the learned PDE.

    N_NN : Neural network that approximates the PDE.

    Collocation_Coords : the collocation points at which we enforce the learned
    PDE. Futher, these should be DISTINCT from the points we test the network at.
    This should be an Nx2 tensor of floats, where N is the number of collocation
    points. The ith row of this tensor should be the coordinates of the ith
    collocation point.

    Data_Coords : A tensor holding the coordinates of the points at which we
    compare the approximate solution to the true one. This should be a Mx2
    tensor of floats, where M is the number of data points. The ith row of this
    tensor should hold the coordinates of the ith data point.

    Data_Values : A tensor holding the value of the true solution at the data
    points. If there are M data points, then this should be an M element
    tensor. The ith element of this tensor should hold the value of the true
    solution at the ith data point.

    optimizer : the optimizer we use to train u_NN and N_NN. It should be
    loaded with the gradients of both networks.

    ----------------------------------------------------------------------------
    returns:
    Nothing! """

    num_Collocation_Points : int = Collocation_Coords.shape[0];
    num_Data_Points        : int = Data_Coords.shape[0];

    # Zero out the gradients in the neural network.
    Optimizer.zero_grad();

    # Evaluate the Loss (Note, we enforce a BC of 0)
    Loss = (Collocation_Loss(u_NN = u_NN, N_NN = N_NN, Collocation_Coords = Collocation_Coords) +
            Data_Loss(u_NN = u_NN, Data_Coords = Data_Coords, Data_Values = Data_Values));

    # Back-propigate to compute gradients of Loss with respect to network
    # weights.
    Loss.backward();

    # update network weights.
    Optimizer.step();



def Testing_Loop(   u_NN : Neural_Network,
                    N_NN : Neural_Network,
                    Collocation_Coords : torch.Tensor,
                    Data_Coords : torch.Tensor,
                    Data_Values : torch.Tensor) -> Tuple[float, float]:
    """ This loop tests the neural network at the specified Collocation and
    Data points. You CAN NOT run this function with no_grad set True.
    Why? Because we need to evaluate derivatives of the solution with respect
    to the inputs! This is a PINN, afterall! Thus, we need torch to build a
    cmputa1tional graph.

    ----------------------------------------------------------------------------
    Arguments:
    u_NN : Neural network that approximates the solution to the learned PDE.

    N_NN : Neural network that approximates the learned PDE.

    Collocation_Coords : the points at which enforce the learned PDE. These
    should be DISTINCT from the collocation points that we use in the training
    loop. This should be an Nx2 tensor of floats, where N is the number of
    collocation points. The ith row of this tensor should be the coordinates of
    the ith collocation point.

    Data_Coords : A tensor holding the coordinates of the points at which we
    compare the approximate solution to the true one. These should be DISTINCT
    from the data points we used in the training loop. This should be a Mx2
    tensor of floats, where M is the number of data points. The ith row of this
    tensor should hold the coordinates of the ith data point.

    ----------------------------------------------------------------------------
    Returns:
    a tuple of floats. The first element holds the collocation loss, while
    the second holds the data loss. """

    # Get the losses at the passed collocation points (Note we enforce a 0 BC)
    Coloc_Loss : float = Collocation_Loss(u_NN = u_NN, N_NN = N_NN, Collocation_Coords = Collocation_Coords).item();
    Data_loss : float = Data_Loss(u_NN = u_NN, Data_Coords = Data_Coords, Data_Values = Data_Values).item();

    # Return the losses.
    return (Coloc_Loss, Data_loss);

    # Should we worry about the computational graph that we build in this
    # function? No.
    # Here's why:
    # Cmputing the losses requires propigating the inputs through the network,
    # thereby building up a computational graph (we need to keep the graph
    # building enabled b/c we have to evaluate derivatives to compute the
    # collocation loss). Normally, these graphs are freed when we call backward.
    # That's what happens in the training loop. Here, we don't use backward.
    # The graphs will be freed, however. This function builds up graphs for
    # Coloc_Loss and Data_loss. When this function returns, however, both
    # variables are freed (along with their graphs!). These graphs do not get
    # passed Coloc_Loss or Data_loss, since both are floats (not Tensors).




# main function!
def main():
    # Load setup data from the setup file.
    Setup_Data = Setup_File_Reader();

    # Test that we got the correct input.
    print("Training PINN with the following parameters:")
    for item in Setup_Data.__dict__.items():
        print(item);

    # Initialize Network hyperparameters.
    Epochs        : int   = Setup_Data.Epochs;
    Learning_Rate : float = Setup_Data.Learning_Rate;


    # Set up the neural network to approximate the PDE solution.
    u_NN = Neural_Network(  Num_Hidden_Layers = Setup_Data.u_Num_Hidden_Layers,
                            Nodes_Per_Layer = Setup_Data.u_Nodes_Per_Layer,
                            Input_Dim = 2,
                            Output_Dim = 1);

    # Set up the neural network to approximate the PDE operator N.
    N_NN = Neural_Network(  Num_Hidden_Layers = Setup_Data.N_Num_Hidden_Layers,
                            Nodes_Per_Layer = Setup_Data.N_Nodes_Per_Layer,
                            Input_Dim = 3,
                            Output_Dim = 1);

    # Pick an optimizer.
    Optimizer = torch.optim.Adam(list(u_NN.parameters()) + list(N_NN.parameters()), lr = Learning_Rate);

    # If we're loading from file, load in the store network's parameters.
    if(     Setup_Data.Load_u_Network_State == True or
            Setup_Data.Load_N_Network_State == True or
            Setup_Data.Load_Optimize_State  == True ):

        # Load the saved checkpoint.
        Checkpoint = torch.load(Setup_Data.Load_File_Name);

        if(Setup_Data.Load_u_Network_State == True):
            u_NN.load_state_dict(Checkpoint["u_Network_State"]);
            u_NN.train();

        if(Setup_Data.Load_N_Network_State == True):
            N_NN.load_state_dict(Checkpoint["N_Network_State"]);
            N_NN.train();

        # Note that this will overwrite the specified Learning Rate using the
        # Learning rate in the saved state. Thus, if this is set to true, then
        # we essentially ignore the learning rate in the setup file.
        if(Setup_Data.Load_Optimize_State == True):
            Optimizer.load_state_dict(Checkpoint["Optimizer_State"]);

    # Set up training and training collocation/boundary points.
    (Train_Coloc_Coords,
     Train_Data_Coords,
     Train_Data_Values,
     Test_Coloc_Coords,
     Test_Data_Coords,
     Test_Data_Values) = Data_Loader("../Data/" + Setup_Data.Data_File_Name, Setup_Data.Num_Training_Points, Setup_Data.Num_Testing_Points);

    # Set up array to hold the testing losses.
    Collocation_Losses = np.empty((Epochs), dtype = np.float);
    Data_Losses        = np.empty((Epochs), dtype = np.float);

    # Loop through the epochs.
    for t in range(Epochs):
        # Run training, testing for this epoch. Log the losses.
        Training_Loop(  u_NN = u_NN,
                        N_NN = N_NN,
                        Collocation_Coords = Train_Coloc_Coords,
                        Data_Coords = Train_Data_Coords,
                        Data_Values = Train_Data_Values,
                        Optimizer = Optimizer);

        (Collocation_Losses[t], Data_Losses[t]) = Testing_Loop( u_NN = u_NN,
                                                                N_NN = N_NN,
                                                                Collocation_Coords = Test_Coloc_Coords,
                                                                Data_Coords = Test_Data_Coords,
                                                                Data_Values = Test_Data_Values );

        # Print losses.
        print(("Epoch #%-4d: " % t), end = '');
        print(("\tCollocation Loss = %7f" % Collocation_Losses[t]), end = '');
        print((",\t Data Loss = %7f" % Data_Losses[t]), end = '');
        print((",\t Total Loss = %7f" % (Collocation_Losses[t] + Data_Losses[t])));

    # Save the network and optimizer states!
    if(Setup_Data.Save_To_File == True):
        torch.save({"u_Network_State" : u_NN.state_dict(),
                    "N_Network_State" : N_NN.state_dict(),
                    "Optimizer_State" : Optimizer.state_dict()},
                    Setup_Data.Save_File_Name);

    exit();

    # Plot final results.
    fig, Axes = Setup_Axes();
    Plotting_Points = Generate_Plot_Gridpoints(50);
    Update_Axes(fig, Axes, u_NN, Plotting_Points, 50);
    plt.show();


if __name__ == '__main__':
    main();
