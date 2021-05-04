import numpy as np;
import torch;
import matplotlib.pyplot as plt;

from Network import Neural_Network;
from Test_Train import Discovery_Testing, Discovery_Training, PINNs_Testing, PINNs_Training;
from Plotter import Update_Axes, Setup_Axes;
from Setup_File_Reader import Setup_File_Reader, Setup_Data_Container;
from Data_Loader import Data_Loader, Data_Container;



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
        print(("Mode is %s while it should be either \"Discovery\" or \"PINNs\"." % Mode));
        print("Something went wrong. Aborting. Thrown by Setup_Optimizer");
        exit();



def main():
    ############################################################################
    # Load setup data from the setup file.
    Setup_Data = Setup_File_Reader();

    # Test that we got the correct input.
    print("Loaded the following settings:");
    for item in Setup_Data.__dict__.items():
        print(item);



    ############################################################################
    # Set up everything.

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
    Data_Container = Data_Loader(
                        Mode = Setup_Data.Mode,
                        Data_File_Path = "../Data/" + Setup_Data.Data_File_Name,
                        Num_Training_Points = Setup_Data.Num_Training_Points,
                        Num_Testing_Points = Setup_Data.Num_Testing_Points);



    ############################################################################
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
                Optimizer                   = Optimizer);

            (IC_Losses[t], BC_Losses[t], Collocation_Losses[t]) = PINNs_Testing(
                u_NN                        = u_NN,
                N_NN                        = N_NN,
                IC_Coords                   = Data_Container.IC_Coords,
                IC_Data                     = Data_Container.IC_Data,
                Lower_Bound_Coords          = Data_Container.Lower_Bound_Coords,
                Upper_Bound_Coords          = Data_Container.Upper_Bound_Coords,
                Periodic_BCs_Highest_Order  = Setup_Data.Periodic_BCs_Highest_Order,
                Collocation_Coords          = Data_Container.Test_Coloc_Coords);

            # Print losses.
            print(("Epoch #%-4d: "              % t)                    , end = '');
            print(("\tIC Loss = %7f"             % IC_Losses[t])         , end = '');
            print(("\tBC Loss = %7f"             % BC_Losses[t])         , end = '');
            print(("\tCollocation Loss = %7f"   % Collocation_Losses[t]), end = '');
            print((",\t Total Loss = %7f"       % (IC_Losses[t] + BC_Losses[t] + Collocation_Losses[t])));
    else:
        print(("Mode is %s while it should be either \"Discovery\" or \"PINNs\"." % Mode));
        print("Something went wrong. Aborting. Thrown by main");
        exit();



    ############################################################################
    # Save the network and optimizer states!

    if(Setup_Data.Save_To_File == True):
        Save_File_Path : str = "../Saves/" + Setup_Data.Save_File_Name;
        torch.save({"u_Network_State" : u_NN.state_dict(),
                    "N_Network_State" : N_NN.state_dict(),
                    "Optimizer_State" : Optimizer.state_dict()},
                    Save_File_Path);



    ############################################################################
    # Plot final results

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
