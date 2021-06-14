import numpy as np;
import torch;
import matplotlib.pyplot as plt;

from Network         import Neural_Network;
from Test_Train      import Discovery_Testing, Discovery_Training, PINNs_Testing, PINNs_Training;
from Extraction      import Generate_Library, Thresholded_Least_Squares, Print_Extracted_PDE, Lasso;
from Plotter         import Update_Axes, Setup_Axes;
from Settings_Reader import Settings_Reader, Settings_Container;
from Data_Setup      import Data_Loader, Data_Container, Generate_Random_Coords;
from Timing          import Timer;



def Setup_Optimizer(
        u_NN            : Neural_Network,
        N_NN            : Neural_Network,
        Mode            : str,
        Learning_Rate   : float,
        Optimizer       : str) -> torch.optim.Optimizer:
    """ This function sets up the optimizer depending on if the N Network has
    learning enabled or not. It also disables gradients for all network
    parameters that are not being learned.

    Note: This function works regardless of how many spatial variables u depends
    on.

    ----------------------------------------------------------------------------
    Arguments:

    u_NN: The network that approximates the PDE solution.

    N_NN: The network that approximates the PDE.

    Mode: Controls which mode the code is running in. You should only call
    this function if mode is either "PINNs" or "Discovery". If "Discovery",
    then we learn both u_NN and N_NN. If "PINNs", then we only learn u_NN
    (and assume that N_NN is trained).

    Learning_Rate: the desired learning rate.

    Optimizer: A string specifies which optimizer we should use.

    ----------------------------------------------------------------------------
    Returns:

    The optimizer! """

    # Construct Params (this depends on the mode).
    if  (Mode == "Discovery"):
        # If we're in discovery mode, then we need to train both u_NN and N_NN.
        # Thus, we need to pass both networks' paramaters to the optimizer.
        Params = list(u_NN.parameters()) + list(N_NN.parameters());
    elif(Mode == "PINNs"):
        # If we're in PINNs mode, then we only need to train u_NN.
        N_NN.requires_grad_(False);
        Params = u_NN.parameters();
    else:
        print(("Mode is %s when it should be either \"Discovery\" or \"PINNs\"." % Mode));
        print("Aborting. Thrown by Setup_Optimizer");
        exit();

    # Now pass Params to the Optimizer.
    if  (Optimizer == "Adam"):
        return torch.optim.Adam(Params, lr = Learning_Rate);
    elif(Optimizer == "LBFGS"):
        return torch.optim.LBFGS(Params, lr = Learning_Rate);
    else:
        print(("Optimizer is %s when it should be \"Adam\" or \"LBFGs\"" % Optimizer));
        print("Aborting. Thrown by Setup_Optimizer");
        exit();



def main():
    ############################################################################
    # Load settings from the settings file.
    Settings = Settings_Reader();

    # Print the settings we read.
    print("Loaded the following settings:");
    for (setting, value) in Settings.__dict__.items():
        print(("%-25s = " % setting) + str(value));



    ############################################################################
    # Set up neural networks, optimizer.

    # Start a timer for program setup.
    Setup_Timer = Timer();
    Setup_Timer.Start();

    # Initialize Network hyperparameters.
    Epochs        : int   = Settings.Epochs;
    Learning_Rate : float = Settings.Learning_Rate;

    # Set up the neural network to approximate the PDE solution.
    u_NN = Neural_Network(  Num_Hidden_Layers   = Settings.u_Num_Hidden_Layers,
                            Neurons_Per_Layer   = Settings.u_Neurons_Per_Layer,
                            Input_Dim           = 2,
                            Output_Dim          = 1,
                            Data_Type           = Settings.Torch_dtype,
                            Activation_Function = Settings.u_Activation_Function);

    # Set up the neural network to approximate the PDE operator N.
    N_NN = Neural_Network(  Num_Hidden_Layers   = Settings.N_Num_Hidden_Layers,
                            Neurons_Per_Layer   = Settings.N_Neurons_Per_Layer,
                            Input_Dim           = Settings.N_Num_u_derivatives + 1,
                            Output_Dim          = 1,
                            Data_Type           = Settings.Torch_dtype,
                            Activation_Function = Settings.N_Activation_Function);

    # Setup the optimizer.
    if(Settings.Mode == "PINNs" or Settings.Mode == "Discovery"):
        Optimizer = Setup_Optimizer(u_NN            = u_NN,
                                    N_NN            = N_NN,
                                    Mode            = Settings.Mode,
                                    Learning_Rate   = Learning_Rate,
                                    Optimizer       = Settings.Optimizer);

    # Check if we're loading anything from file.
    if(     Settings.Load_u_Network_State == True or
            Settings.Load_N_Network_State == True or
            Settings.Load_Optimize_State  == True):

        # Load the saved checkpoint.
        Load_File_Path : str = "../Saves/" + Settings.Load_File_Name;
        Saved_State = torch.load(Load_File_Path);

        if(Settings.Load_u_Network_State == True):
            u_NN.load_state_dict(Saved_State["u_Network_State"]);
            u_NN.train();

        if(Settings.Load_N_Network_State == True):
            N_NN.load_state_dict(Saved_State["N_Network_State"]);
            N_NN.train();

        # Note that this will overwrite the Learning Rate using the
        # Learning rate in the saved state. Thus, if this is set to true, then
        # we essentially ignore the learning rate in the settings.
        if(Settings.Load_Optimize_State  == True):
            Optimizer.load_state_dict(Saved_State["Optimizer_State"]);



    ############################################################################
    # Set up points (Data, IC, BC, Collocation, Extraction).

    # If we're in Discovery mode, this will set up the testing and training
    # data points and values. If we're in PINNs mode, this will also set up IC
    # and BC points. This should also give us the upper and lower bounds for the
    # domain.
    Data_Container = Data_Loader(Settings);

    # Set up mode specific points.
    if  (Settings.Mode == "PINNs" or Settings.Mode == "Discovery"):
        # In these modes, we need to set up the Collocation points.

        # Generate Collocation points.
        Data_Container.Train_Colloc_Coords = Generate_Random_Coords(
                Dim_Lower_Bounds    = Data_Container.Dim_Lower_Bounds,
                Dim_Upper_Bounds    = Data_Container.Dim_Upper_Bounds,
                Num_Points          = Settings.Num_Train_Colloc_Points,
                Data_Type           = Settings.Torch_dtype);

        Data_Container.Test_Colloc_Coords = Generate_Random_Coords(
                Dim_Lower_Bounds    = Data_Container.Dim_Lower_Bounds,
                Dim_Upper_Bounds    = Data_Container.Dim_Upper_Bounds,
                Num_Points          = Settings.Num_Test_Colloc_Points,
                Data_Type           = Settings.Torch_dtype);

    elif(Settings.Mode == "Extraction"):
        # In this mode we need to set up the Extraction points.

        # Generate Collocation points.
        Data_Container.Extraction_Coords = Generate_Random_Coords(
                Dim_Lower_Bounds    = Data_Container.Dim_Lower_Bounds,
                Dim_Upper_Bounds    = Data_Container.Dim_Upper_Bounds,
                Num_Points          = Settings.Num_Extraction_Points,
                Data_Type           = Settings.Torch_dtype);


    # Setup is done! Figure out how long it took.
    Setup_Time : float = Setup_Timer.Stop();
    print("Setup took %fs." % Setup_Time);



    ############################################################################
    # Loop through the epochs.

    # Start a timer for the Epochs.
    Main_Timer = Timer();
    Main_Timer.Start();

    if  (Settings.Mode == "PINNs"):
        # Setup arrays for the different kinds of losses.
        IC_Losses          = np.empty((Epochs), dtype = Settings.Numpy_dtype);
        BC_Losses          = np.empty((Epochs), dtype = Settings.Numpy_dtype);
        Collocation_Losses = np.empty((Epochs), dtype = Settings.Numpy_dtype);

        for t in range(Epochs):
            PINNs_Training(
                u_NN                        = u_NN,
                N_NN                        = N_NN,
                IC_Coords                   = Data_Container.IC_Coords,
                IC_Data                     = Data_Container.IC_Data,
                Lower_Bound_Coords          = Data_Container.Lower_Bound_Coords,
                Upper_Bound_Coords          = Data_Container.Upper_Bound_Coords,
                Periodic_BCs_Highest_Order  = Settings.Periodic_BCs_Highest_Order,
                Collocation_Coords          = Data_Container.Train_Colloc_Coords,
                Optimizer                   = Optimizer);

            (IC_Losses[t], BC_Losses[t], Collocation_Losses[t]) = PINNs_Testing(
                u_NN                        = u_NN,
                N_NN                        = N_NN,
                IC_Coords                   = Data_Container.IC_Coords,
                IC_Data                     = Data_Container.IC_Data,
                Lower_Bound_Coords          = Data_Container.Lower_Bound_Coords,
                Upper_Bound_Coords          = Data_Container.Upper_Bound_Coords,
                Periodic_BCs_Highest_Order  = Settings.Periodic_BCs_Highest_Order,
                Collocation_Coords          = Data_Container.Test_Colloc_Coords);

            # Print losses.
            print(("Epoch #%-4d: "             % t)                    , end = '');
            print(("\tIC Loss = %.7f"          % IC_Losses[t])         , end = '');
            print(("\tBC Loss = %.7f"          % BC_Losses[t])         , end = '');
            print(("\tCollocation Loss = %.7f" % Collocation_Losses[t]), end = '');
            print((",\t Total Loss = %.7f"     % (IC_Losses[t] + BC_Losses[t] + Collocation_Losses[t])));

    elif(Settings.Mode == "Discovery"):
        # Set up array for the different kinds of losses.
        Collocation_Losses = np.empty((Epochs), dtype = Settings.Numpy_dtype);
        Data_Losses        = np.empty((Epochs), dtype = Settings.Numpy_dtype);

        for t in range(Epochs):
            Discovery_Training(
                u_NN                = u_NN,
                N_NN                = N_NN,
                Collocation_Coords  = Data_Container.Train_Colloc_Coords,
                Data_Coords         = Data_Container.Train_Data_Coords,
                Data_Values         = Data_Container.Train_Data_Values,
                Optimizer           = Optimizer);

            (Collocation_Losses[t], Data_Losses[t]) = Discovery_Testing(
                u_NN                = u_NN,
                N_NN                = N_NN,
                Collocation_Coords  = Data_Container.Test_Colloc_Coords,
                Data_Coords         = Data_Container.Test_Data_Coords,
                Data_Values         = Data_Container.Test_Data_Values );

            # Print losses.
            print(("Epoch #%-4d: "              % t)                    , end = '');
            print(("\tCollocation Loss = %.7f"   % Collocation_Losses[t]), end = '');
            print((",\t Data Loss = %.7f"        % Data_Losses[t])       , end = '');
            print((",\t Total Loss = %.7f"       % (Collocation_Losses[t] + Data_Losses[t])));

    elif(Settings.Mode == "Extraction"):
        # Generate the library!
        (N_NN_batch,
         Library,
         num_multi_indices,
         multi_indices_list) = Generate_Library(
                                    u_NN            = u_NN,
                                    N_NN            = N_NN,
                                    Coords          = Data_Container.Extraction_Coords,
                                    num_derivatives = Settings.N_Num_u_derivatives,
                                    Poly_Degree     = Settings.Extracted_term_degree);

        #Extracted_PDE = Lasso(
        #                    A         = Library,
        #                    b         = N_NN_batch,
        #                    alpha     = Settings.Least_Squares_Threshold);

        Extracted_PDE = Thresholded_Least_Squares(
                            A         = Library,
                            b         = N_NN_batch,
                            threshold = Settings.Least_Squares_Threshold);

        Print_Extracted_PDE(
            Extracted_PDE      = Extracted_PDE,
            num_multi_indices  = num_multi_indices,
            multi_indices_list = multi_indices_list);

    else:
        print(("Mode is %s while it should be either \"PINNs\", \"Discovery\", or \"Extraction\"." % Mode));
        print("Something went wrong. Aborting. Thrown by main.");
        exit();


    # Epochs are done. Figure out how long they took!
    Main_Time = Main_Timer.Stop();

    if (Settings.Mode == "PINNs" or Settings.Mode == "Discovery"):
        print("Running %d epochs took %fs." % (Epochs, Main_Time));
        print("That's an average of %fs per epoch!" % (Main_Time/Epochs));

    elif(Settings.Mode == "Extraction"):
        print("Extraction took %fs." % Main_Time);



    ############################################################################
    # Save the network and optimizer states!
    # This only makes sense if we're in PINNs or Discovery modes.

    if((Settings.Mode == "PINNs" or Settings.Mode == "Discovery") and Settings.Save_To_File == True):
        Save_File_Path : str = "../Saves/" + Settings.Save_File_Name;
        torch.save({"u_Network_State" : u_NN.state_dict(),
                    "N_Network_State" : N_NN.state_dict(),
                    "Optimizer_State" : Optimizer.state_dict()},
                    Save_File_Path);



    ############################################################################
    # Plot final results

    if(Settings.Plot_Final_Results == True):
        fig, Axes = Setup_Axes();
        Update_Axes(fig                 = fig,
                    Axes                = Axes,
                    u_NN                = u_NN,
                    N_NN                = N_NN,
                    x_points            = Data_Container.x_points,
                    t_points            = Data_Container.t_points,
                    True_Sol_On_Grid    = Data_Container.True_Sol);
        plt.show();


if __name__ == '__main__':
    main();
