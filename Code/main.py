import numpy as np;
import torch;
import matplotlib.pyplot as plt;

from Network         import Neural_Network;
from Test_Train      import Discovery_Testing, Discovery_Training, PINNs_Testing, PINNs_Training;
from Extraction      import Generate_Library, Thresholded_Least_Squares, Print_Extracted_PDE, Lasso_Selection;
from Plotter         import Initialize_Axes, Setup_Axes;
from Settings_Reader import Settings_Reader, Settings_Container;
from Data_Setup      import Data_Loader, Data_Container, Generate_Random_Coords;
from Timing          import Timer;



def main():
    ############################################################################
    # Load settings from the settings file, print them.
    Settings = Settings_Reader();
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
    Sol_NN = Neural_Network( Num_Hidden_Layers   = Settings.Sol_Num_Hidden_Layers,
                             Neurons_Per_Layer   = Settings.Sol_Neurons_Per_Layer,
                             Input_Dim           = 2,
                             Output_Dim          = 1,
                             Data_Type           = Settings.Torch_dtype,
                             Device              = Settings.Device,
                             Activation_Function = Settings.Sol_Activation_Function,
                             Dropout_Probability = Settings.Sol_Dropout_Probability);

    # Set up the neural network to approximate the PDE operator N.
    PDE_NN = Neural_Network( Num_Hidden_Layers   = Settings.PDE_Num_Hidden_Layers,
                             Neurons_Per_Layer   = Settings.PDE_Neurons_Per_Layer,
                             Input_Dim           = Settings.PDE_Num_Sol_derivatives + 1,
                             Output_Dim          = 1,
                             Data_Type           = Settings.Torch_dtype,
                             Device              = Settings.Device,
                             Activation_Function = Settings.PDE_Activation_Function,
                             Dropout_Probability = Settings.PDE_Dropout_Probability);

    # Setup the optimizer.
    Optimizer = None;
    if(Settings.Mode == "PINNs" or Settings.Mode == "Discovery"):
        # Construct Params (this depends on the mode).
        if  (Settings.Mode == "Discovery"):
            # If we're in discovery mode, then we need to train both Sol_NN and
            # PDE_NN. Thus, we need to pass both networks' paramaters to the
            # optimizer.
            Params = list(Sol_NN.parameters()) + list(PDE_NN.parameters());
        elif(Settings.Mode == "PINNs"):
            # If we're in PINNs mode, then we only need to train Sol_NN.
            PDE_NN.requires_grad_(False);
            Params = Sol_NN.parameters();

        # Now pass Params to the Optimizer.
        if  (Settings.Optimizer == "Adam"):
            Optimizer = torch.optim.Adam(Params, lr = Learning_Rate);
        elif(Settings.Optimizer == "LBFGS"):
            Optimizer = torch.optim.LBFGS(Params, lr = Learning_Rate);
        else:
            print(("Optimizer is %s when it should be \"Adam\" or \"LBFGS\"" % Optimizer));
            print("Aborting. Thrown by Setup_Optimizer");
            exit();

    # Check if we're loading anything from file.
    if( Settings.Load_Sol_Network_State == True or
        Settings.Load_PDE_Network_State == True or
        Settings.Load_Optimize_State    == True):

        # Load the saved checkpoint. Make sure to map it to the correct device.
        Load_File_Path : str = "../Saves/" + Settings.Load_File_Name;
        Saved_State = torch.load(Load_File_Path, map_location = Settings.Device);

        if(Settings.Load_Sol_Network_State == True):
            Sol_NN.load_state_dict(Saved_State["Sol_Network_State"]);

        if(Settings.Load_PDE_Network_State == True):
            PDE_NN.load_state_dict(Saved_State["PDE_Network_State"]);

        if(Settings.Load_Optimize_State  == True):
            Optimizer.load_state_dict(Saved_State["Optimizer_State"]);

            # Enforce the new learning rate (do not use the saved one).
            for param_group in Optimizer.param_groups:
                param_group['lr'] = Settings.Learning_Rate;


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
                Dim_Lower_Bounds = Data_Container.Dim_Lower_Bounds,
                Dim_Upper_Bounds = Data_Container.Dim_Upper_Bounds,
                Num_Points       = Settings.Num_Train_Colloc_Points,
                Data_Type        = Settings.Torch_dtype,
                Device           = Settings.Device);

        Data_Container.Test_Colloc_Coords = Generate_Random_Coords(
                Dim_Lower_Bounds = Data_Container.Dim_Lower_Bounds,
                Dim_Upper_Bounds = Data_Container.Dim_Upper_Bounds,
                Num_Points       = Settings.Num_Test_Colloc_Points,
                Data_Type        = Settings.Torch_dtype,
                Device           = Settings.Device);

    elif(Settings.Mode == "Extraction"):
        # In Extraction mode, we need to set up the Extraction points.

        # Generate Collocation points.
        Data_Container.Extraction_Coords = Generate_Random_Coords(
                Dim_Lower_Bounds    = Data_Container.Dim_Lower_Bounds,
                Dim_Upper_Bounds    = Data_Container.Dim_Upper_Bounds,
                Num_Points          = Settings.Num_Extraction_Points,
                Data_Type           = Settings.Torch_dtype,
                Device              = Settings.Device);


    # Setup is done! Figure out how long it took.
    Setup_Time : float = Setup_Timer.Stop();
    print("Setup took %fs." % Setup_Time);



    ############################################################################
    # Loop through the epochs.

    # Start a timer for the Epochs.
    Main_Timer = Timer();
    Main_Timer.Start();

    if  (Settings.Mode == "PINNs"):
        if(Epochs != 0):
            # Set up array for the different losses. We only print the losses
            # every few Epochs. As a result, the loss arrays only need
            # (Epochs - 2)//Epochs_Between_Prints + 2 rows (think about it).
            Epochs_Between_Prints : int = 10;
            Num_Loss_Measurements : int = (Epochs - 2)//Epochs_Between_Prints + 2;
            Test_IC_Loss    = np.empty(Num_Loss_Measurements, dtype = Settings.Numpy_dtype);
            Test_BC_Loss    = np.empty(Num_Loss_Measurements, dtype = Settings.Numpy_dtype);
            Test_Data_Loss  = np.empty(Num_Loss_Measurements, dtype = Settings.Numpy_dtype);
            Train_IC_Loss   = np.empty(Num_Loss_Measurements, dtype = Settings.Numpy_dtype);
            Train_BC_Loss   = np.empty(Num_Loss_Measurements, dtype = Settings.Numpy_dtype);
            Train_Data_Loss = np.empty(Num_Loss_Measurements, dtype = Settings.Numpy_dtype);

            Loss_Counter : int = 0;

        for t in range(Epochs):
            PINNs_Training(
                Sol_NN                      = Sol_NN,
                PDE_NN                      = PDE_NN,
                IC_Coords                   = Data_Container.IC_Coords,
                IC_Data                     = Data_Container.IC_Data,
                Lower_Bound_Coords          = Data_Container.Lower_Bound_Coords,
                Upper_Bound_Coords          = Data_Container.Upper_Bound_Coords,
                Periodic_BCs_Highest_Order  = Settings.Periodic_BCs_Highest_Order,
                Collocation_Coords          = Data_Container.Train_Colloc_Coords,
                Optimizer                   = Optimizer,
                Data_Type                   = Settings.Torch_dtype,
                Device                      = Settings.Device);

            # Periodically print loss updates. Otherwise, just print the Epoch #
            # to indicate that we're still alive.
            if((t % Epochs_Between_Prints == 0) or t == Epochs - 1):
                # Alias the Loss counter for brevity
                i : int = Loss_Counter;

                (Test_IC_Loss[i], Test_BC_Loss[i], Test_Data_Loss[i]) = PINNs_Testing(
                    Sol_NN                      = Sol_NN,
                    PDE_NN                      = PDE_NN,
                    IC_Coords                   = Data_Container.IC_Coords,
                    IC_Data                     = Data_Container.IC_Data,
                    Lower_Bound_Coords          = Data_Container.Lower_Bound_Coords,
                    Upper_Bound_Coords          = Data_Container.Upper_Bound_Coords,
                    Periodic_BCs_Highest_Order  = Settings.Periodic_BCs_Highest_Order,
                    Collocation_Coords          = Data_Container.Test_Colloc_Coords,
                    Data_Type                   = Settings.Torch_dtype,
                    Device                      = Settings.Device);

                (Train_IC_Loss[i], Train_BC_Loss[i], Train_Data_Loss[i]) = PINNs_Testing(
                    Sol_NN                      = Sol_NN,
                    PDE_NN                      = PDE_NN,
                    IC_Coords                   = Data_Container.IC_Coords,
                    IC_Data                     = Data_Container.IC_Data,
                    Lower_Bound_Coords          = Data_Container.Lower_Bound_Coords,
                    Upper_Bound_Coords          = Data_Container.Upper_Bound_Coords,
                    Periodic_BCs_Highest_Order  = Settings.Periodic_BCs_Highest_Order,
                    Collocation_Coords          = Data_Container.Train_Colloc_Coords,
                    Data_Type                   = Settings.Torch_dtype,
                    Device                      = Settings.Device);

                # Print losses!
                print("Epoch #%-4d | Test: \t IC = %.7f\t BC = %.7f\t Collocation = %.7f\t Total = %.7f"
                      % (t, Test_IC_Loss[i], Test_BC_Loss[i], Test_Data_Loss[i],
                         Test_IC_Loss[i] + Test_BC_Loss[i] + Test_Data_Loss[i]));
                print("            | Train:\t IC = %.7f\t BC = %.7f\t Collocation = %.7f\t Total = %.7f"
                      % (Train_IC_Loss[i], Train_BC_Loss[i], Train_Data_Loss[i],
                         Train_IC_Loss[i] + Train_BC_Loss[i] + Train_Data_Loss[i]));

                # Increment the counter.
                Loss_Counter += 1;
            else:
                print(("Epoch #%-4d | "   % t));

    elif(Settings.Mode == "Discovery"):
        if(Epochs != 0):
            # Set up arrays for the different losses. We only measure the loss every
            # few epochs. As a result, the loss arrays only need
            # (Epochs - 2)//Epochs_Between_Prints + 2 rows (think about it).
            Epochs_Between_Prints : int = 10;
            Num_Loss_Measurements : int = (Epochs - 2)//Epochs_Between_Prints + 2;
            Test_Coll_Loss  = np.empty(Num_Loss_Measurements, dtype = Settings.Numpy_dtype);
            Test_Data_Loss  = np.empty(Num_Loss_Measurements, dtype = Settings.Numpy_dtype);
            Train_Coll_Loss = np.empty(Num_Loss_Measurements, dtype = Settings.Numpy_dtype);
            Train_Data_Loss = np.empty(Num_Loss_Measurements, dtype = Settings.Numpy_dtype);

            Loss_Counter : int = 0;

        for t in range(Epochs):
            Discovery_Training(
                Sol_NN              = Sol_NN,
                PDE_NN              = PDE_NN,
                Collocation_Coords  = Data_Container.Train_Colloc_Coords,
                Data_Coords         = Data_Container.Train_Data_Coords,
                Data_Values         = Data_Container.Train_Data_Values,
                Optimizer           = Optimizer,
                Data_Type           = Settings.Torch_dtype,
                Device              = Settings.Device);

            # Periodically print loss updates. Otherwise, just print the Epoch #
            # to indicate that we're still alive.
            if(t % Epochs_Between_Prints == 0 or t == Epochs - 1):
                # Alias the Loss counter for brevity
                i : int = Loss_Counter;

                # Evaluate losses on Testing, Training points.
                (Test_Coll_Loss[i], Test_Data_Loss[i]) = Discovery_Testing(
                    Sol_NN              = Sol_NN,
                    PDE_NN              = PDE_NN,
                    Collocation_Coords  = Data_Container.Test_Colloc_Coords,
                    Data_Coords         = Data_Container.Test_Data_Coords,
                    Data_Values         = Data_Container.Test_Data_Values,
                    Data_Type           = Settings.Torch_dtype,
                    Device              = Settings.Device);

                (Train_Coll_Loss[i], Test_Data_Loss[i]) = Discovery_Testing(
                    Sol_NN              = Sol_NN,
                    PDE_NN              = PDE_NN,
                    Collocation_Coords  = Data_Container.Train_Colloc_Coords,
                    Data_Coords         = Data_Container.Train_Data_Coords,
                    Data_Values         = Data_Container.Train_Data_Values,
                    Data_Type           = Settings.Torch_dtype,
                    Device              = Settings.Device);

                # Print losses!
                print("Epoch #%-4d | Test: \t Collocation = %.7f\t Data = %.7f\t Total = %.7f"
                    % (t, Test_Coll_Loss[i], Test_Data_Loss[i], Test_Coll_Loss[i] + Test_Data_Loss[i]));
                print("            | Train:\t Collocation = %.7f\t Data = %.7f\t Total = %.7f"
                    % (Train_Coll_Loss[i], Test_Data_Loss[i], Train_Coll_Loss[i] + Test_Data_Loss[i]));

                # Increment the counter.
                Loss_Counter += 1;
            else:
                print(("Epoch #%-4d | "   % t));

    elif(Settings.Mode == "Extraction"):
        # Generate the library!
        (PDE_NN_batch,
         Library,
         num_multi_indices,
         multi_indices_list) = Generate_Library(
                                    Sol_NN          = Sol_NN,
                                    PDE_NN          = PDE_NN,
                                    Coords          = Data_Container.Extraction_Coords,
                                    num_derivatives = Settings.PDE_Num_Sol_derivatives,
                                    Poly_Degree     = Settings.Extracted_term_degree,
                                    Torch_Data_Type = Settings.Torch_dtype,
                                    Device          = Settings.Device);

        #Extracted_PDE = Lasso_Selection(
        #                    A         = Library,
        #                    b         = PDE_NN_batch,
        #                    alpha     = Settings.Least_Squares_Threshold);

        Extracted_PDE = Thresholded_Least_Squares(
                            A         = Library,
                            b         = PDE_NN_batch,
                            threshold = Settings.Least_Squares_Threshold);

        Print_Extracted_PDE(
            Extracted_PDE      = Extracted_PDE,
            num_multi_indices  = num_multi_indices,
            multi_indices_list = multi_indices_list);

    else:
        print(("Mode is %s. It should be one of \"PINNs\", \"Discovery\", \"Extraction\"." % Settings.Mode));
        print("Something went wrong. Aborting. Thrown by main.");
        exit();


    # Epochs are done. Figure out how long they took!
    Main_Time = Main_Timer.Stop();

    if (Settings.Mode == "PINNs" or Settings.Mode == "Discovery"):
        print("Running %d epochs took %fs." % (Epochs, Main_Time));
        if(Epochs > 0):
            print("That's an average of %fs per epoch!" % (Main_Time/Epochs));

    elif(Settings.Mode == "Extraction"):
        print("Extraction took %fs." % Main_Time);



    ############################################################################
    # Save the network and optimizer states!
    # This only makes sense if we're in PINNs or Discovery modes since those
    # those modes actually train something.

    if((Settings.Mode == "PINNs" or Settings.Mode == "Discovery") and Settings.Save_To_File == True):
        Save_File_Path : str = "../Saves/" + Settings.Save_File_Name;
        torch.save({"Sol_Network_State" : Sol_NN.state_dict(),
                    "PDE_Network_State" : PDE_NN.state_dict(),
                    "Optimizer_State" : Optimizer.state_dict()},
                    Save_File_Path);



    ############################################################################
    # Plot final results

    if(Settings.Plot_Final_Results == True or Settings.Save_Plot == True):
        # Make note of how long this takes.
        Print_Timer = Timer();
        Print_Timer.Start();
        print("Plotting... ", end = '');

        # Now, setup the plot.
        fig, Axes = Initialize_Axes();
        Setup_Axes(fig              = fig,
                   Axes             = Axes,
                   Sol_NN           = Sol_NN,
                   PDE_NN           = PDE_NN,
                   x_points         = Data_Container.x_points,
                   t_points         = Data_Container.t_points,
                   True_Sol_On_Grid = Data_Container.True_Sol,
                   Torch_dtype      = Settings.Torch_dtype,
                   Device           = Settings.Device);
        Print_Time = Print_Timer.Stop();
        print("Done! Took %fs" % Print_Time);

        # Show the plot (if we're supposed to)
        if(Settings.Plot_Final_Results == True):
            plt.show();

        # Save the plot (if we're supposed to)
        if(Settings.Save_Plot == True):
            fig.savefig(fname = "../Figures/%s" % Settings.Save_File_Name);


if __name__ == '__main__':
    main();
