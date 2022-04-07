import numpy;
import torch;

from Network         import Neural_Network;
from Test_Train      import Discovery_Testing, Discovery_Training, PINNs_Testing, PINNs_Training;
from Extraction      import Generate_Library, Print_Extracted_PDE, Recursive_Feature_Elimination, Rank_Candidate_Solutions;
from Settings_Reader import Settings_Reader, Settings_Container;
from Data_Setup      import Data_Loader, Data_Container, Generate_Random_Coords;
from Timing          import Timer;



def main():
    ############################################################################
    # Load settings, print them.

    Settings = Settings_Reader();
    print("Loaded the following settings:");
    for (setting, value) in Settings.__dict__.items():
        print(("%-30s = " % setting) + str(value));



    ############################################################################
    # Set up neural networks, optimizer.

    # Start a timer for program setup.
    Setup_Timer = Timer();
    Setup_Timer.Start();

    # Set Network hyperparameters.
    Epochs        : int   = Settings.Epochs;
    Learning_Rate : float = Settings.Learning_Rate;

    # Initialize the Solution, PDE networks.
    Sol_NN = Neural_Network( Num_Hidden_Layers   = Settings.Sol_Num_Hidden_Layers,
                             Neurons_Per_Layer   = Settings.Sol_Neurons_Per_Layer,
                             Input_Dim           = 1,
                             Output_Dim          = 1,
                             Data_Type           = torch.float32,
                             Device              = Settings.Device,
                             Activation_Function = Settings.Sol_Activation_Function,
                             Batch_Norm          = False);

    PDE_NN = Neural_Network( Num_Hidden_Layers   = Settings.PDE_Num_Hidden_Layers,
                             Neurons_Per_Layer   = Settings.PDE_Neurons_Per_Layer,
                             Input_Dim           = Settings.PDE_Spatial_Derivative_Order + 1,
                             Output_Dim          = 1,
                             Data_Type           = torch.float32,
                             Device              = Settings.Device,
                             Activation_Function = Settings.PDE_Activation_Function,
                             Batch_Norm          = Settings.PDE_Normalize_Inputs);

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
            Optimizer = torch.optim.Adam(   Params, lr = Learning_Rate);
        elif(Settings.Optimizer == "LBFGS"):
            Optimizer = torch.optim.LBFGS(  Params, lr = Learning_Rate);
        elif(Settings.Optimizer == "SGD"):
            Optimizer = torch.optim.SGD(    Params, lr = Learning_Rate, momentum = 0.9, nesterov = True);
        else:
            print(("Optimizer is %s when it should be \"Adam\", \"LBFGS\", or \"SGD\"" % Settings.Optimizer));
            print("Aborting.");
            exit();

    # Check if we're loading anything from file.
    if( Settings.Load_Sol_Network_State == True or
        Settings.Load_PDE_Network_State == True or
        Settings.Load_Optimizer_State   == True):

        # Load the saved checkpoint. Make sure to map it to the correct device.
        Load_File_Path : str = "../Saves/" + Settings.Load_File_Name;
        Saved_State = torch.load(Load_File_Path, map_location = Settings.Device);

        if(Settings.Load_Sol_Network_State == True):
            Sol_NN.load_state_dict(Saved_State["Sol_Network_State"]);

        if(Settings.Load_PDE_Network_State == True):
            PDE_NN.load_state_dict(Saved_State["PDE_Network_State"]);

        # We do not load the optimizier if we're in Extraction mode.
        if(Settings.Load_Optimizer_State == True and Settings.Mode != "Extraction"):
            Optimizer.load_state_dict(Saved_State["Optimizer_State"]);

            # Enforce the new learning rate (do not use the saved one).
            for param_group in Optimizer.param_groups:
                param_group['lr'] = Settings.Learning_Rate;


    ############################################################################
    # Set up Data
    # If we're in Discovery mode, this will set up the testing and training
    # data points and values. If we're in PINNs mode, this will set up IC and
    # BC points. This should also give us the upper and lower bounds for the
    # domain.
    Data_Container = Data_Loader(   DataSet_Name    = Settings.DataSet_Name,
                                    Device          = Settings.Device,
                                    Mode            = Settings.Mode);

    # Setup is done! Figure out how long it took.
    Setup_Time : float = Setup_Timer.Stop();
    print("Setup took %fs." % Setup_Time);



    ############################################################################
    # Epochs, Extraction

    # Start a timer for the Epochs.
    Main_Timer = Timer();
    Main_Timer.Start();

    if  (Settings.Mode == "PINNs"):
        # Setup Loss tracking.
        if(Epochs != 0):
            # Set up array for the different losses. We only print the losses
            # every few Epochs. As a result, the loss arrays only need
            # (Epochs - 2)//Epochs_Between_Prints + 2 rows (think about it).
            Num_Loss_Measurements : int = (Epochs - 2)//Settings.Epochs_Between_Prints + 2;
            Test_IC_Loss    = numpy.empty(Num_Loss_Measurements, dtype = numpy.float32);
            Test_BC_Loss    = numpy.empty(Num_Loss_Measurements, dtype = numpy.float32);
            Test_Data_Loss  = numpy.empty(Num_Loss_Measurements, dtype = numpy.float32);
            Train_IC_Loss   = numpy.empty(Num_Loss_Measurements, dtype = numpy.float32);
            Train_BC_Loss   = numpy.empty(Num_Loss_Measurements, dtype = numpy.float32);
            Train_Data_Loss = numpy.empty(Num_Loss_Measurements, dtype = numpy.float32);

            Loss_Counter : int = 0;

        for t in range(Epochs):
            # Check if we should generate new Collocation points.
            if(t % Settings.Epochs_For_New_Coll_Pts == 0):
                Train_Colloc_Coords = Generate_Random_Coords(
                        Dim_Lower_Bounds = Data_Container.Dim_Lower_Bounds,
                        Dim_Upper_Bounds = Data_Container.Dim_Upper_Bounds,
                        Num_Points       = Settings.Num_Train_Colloc_Points,
                        Data_Type        = torch.float32,
                        Device           = Settings.Device);

                Test_Colloc_Coords = Generate_Random_Coords(
                        Dim_Lower_Bounds = Data_Container.Dim_Lower_Bounds,
                        Dim_Upper_Bounds = Data_Container.Dim_Upper_Bounds,
                        Num_Points       = Settings.Num_Test_Colloc_Points,
                        Data_Type        = torch.float32,
                        Device           = Settings.Device);

            PINNs_Training(
                Sol_NN                      = Sol_NN,
                PDE_NN                      = PDE_NN,
                Time_Derivative_Order       = Settings.PDE_Time_Derivative_Order,
                Spatial_Derivative_Order    = Settings.PDE_Spatial_Derivative_Order,
                IC_Coords                   = Data_Container.IC_Coords,
                IC_Data                     = Data_Container.IC_Data,
                Lower_Bound_Coords          = Data_Container.Lower_Bound_Coords,
                Upper_Bound_Coords          = Data_Container.Upper_Bound_Coords,
                Periodic_BCs_Highest_Order  = Settings.Periodic_BCs_Highest_Order,
                Collocation_Coords          = Train_Colloc_Coords,
                Optimizer                   = Optimizer,
                Data_Type                   = torch.float32,
                Device                      = Settings.Device);

            # Periodically print loss updates. In all other Epochs, print the
            # epoch number to indiciate that the code is still running.
            if((t % Settings.Epochs_Between_Prints == 0) or t == Epochs - 1):
                # Alias the Loss counter for brevity
                i : int = Loss_Counter;

                # Determine losses on the testing, training data.
                (Test_IC_Loss[i], Test_BC_Loss[i], Test_Data_Loss[i]) = PINNs_Testing(
                    Sol_NN                      = Sol_NN,
                    PDE_NN                      = PDE_NN,
                    Time_Derivative_Order       = Settings.PDE_Time_Derivative_Order,
                    Spatial_Derivative_Order    = Settings.PDE_Spatial_Derivative_Order,
                    IC_Coords                   = Data_Container.IC_Coords,
                    IC_Data                     = Data_Container.IC_Data,
                    Lower_Bound_Coords          = Data_Container.Lower_Bound_Coords,
                    Upper_Bound_Coords          = Data_Container.Upper_Bound_Coords,
                    Periodic_BCs_Highest_Order  = Settings.Periodic_BCs_Highest_Order,
                    Collocation_Coords          = Test_Colloc_Coords,
                    Data_Type                   = torch.float32,
                    Device                      = Settings.Device);

                (Train_IC_Loss[i], Train_BC_Loss[i], Train_Data_Loss[i]) = PINNs_Testing(
                    Sol_NN                      = Sol_NN,
                    PDE_NN                      = PDE_NN,
                    Time_Derivative_Order       = Settings.PDE_Time_Derivative_Order,
                    Spatial_Derivative_Order    = Settings.PDE_Spatial_Derivative_Order,
                    IC_Coords                   = Data_Container.IC_Coords,
                    IC_Data                     = Data_Container.IC_Data,
                    Lower_Bound_Coords          = Data_Container.Lower_Bound_Coords,
                    Upper_Bound_Coords          = Data_Container.Upper_Bound_Coords,
                    Periodic_BCs_Highest_Order  = Settings.Periodic_BCs_Highest_Order,
                    Collocation_Coords          = Train_Colloc_Coords,
                    Data_Type                   = torch.float32,
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
        # Setup Loss tracking.
        if(Epochs != 0):
            # Set up arrays for the different losses. We only measure the loss every
            # few epochs. As a result, the loss arrays only need
            # (Epochs - 2)//Epochs_Between_Prints + 2 rows (think about it).
            Num_Loss_Measurements : int = (Epochs - 2)//Settings.Epochs_Between_Prints + 2;
            Test_Coll_Loss  = numpy.empty(Num_Loss_Measurements, dtype = numpy.float32);
            Test_Data_Loss  = numpy.empty(Num_Loss_Measurements, dtype = numpy.float32);
            Train_Coll_Loss = numpy.empty(Num_Loss_Measurements, dtype = numpy.float32);
            Train_Data_Loss = numpy.empty(Num_Loss_Measurements, dtype = numpy.float32);

            Loss_Counter : int = 0;

        for t in range(Epochs):
            # Check if we should generate new Collocation points.
            if(t % Settings.Epochs_For_New_Coll_Pts == 0):
                Train_Colloc_Coords = Generate_Random_Coords(
                        Dim_Lower_Bounds = Data_Container.Dim_Lower_Bounds,
                        Dim_Upper_Bounds = Data_Container.Dim_Upper_Bounds,
                        Num_Points       = Settings.Num_Train_Colloc_Points,
                        Data_Type        = torch.float32,
                        Device           = Settings.Device);

                Test_Colloc_Coords = Generate_Random_Coords(
                        Dim_Lower_Bounds = Data_Container.Dim_Lower_Bounds,
                        Dim_Upper_Bounds = Data_Container.Dim_Upper_Bounds,
                        Num_Points       = Settings.Num_Test_Colloc_Points,
                        Data_Type        = torch.float32,
                        Device           = Settings.Device);

            # Now train!
            Discovery_Training(
                Sol_NN                      = Sol_NN,
                PDE_NN                      = PDE_NN,
                Time_Derivative_Order       = Settings.PDE_Time_Derivative_Order,
                Spatial_Derivative_Order    = Settings.PDE_Spatial_Derivative_Order,
                Collocation_Coords          = Train_Colloc_Coords,
                Data_Coords                 = Data_Container.Train_Data_Coords,
                Data_Values                 = Data_Container.Train_Data_Values,
                Optimizer                   = Optimizer,
                Data_Type                   = torch.float32,
                Device                      = Settings.Device);

            # Periodically print loss updates. Otherwise, just print the Epoch #
            # to indicate that we're still alive.
            if(t % Settings.Epochs_Between_Prints == 0 or t == Epochs - 1):
                # Alias the Loss counter for brevity
                i : int = Loss_Counter;

                # Evaluate losses on Testing, Training points.
                (Test_Coll_Loss[i], Test_Data_Loss[i]) = Discovery_Testing(
                    Sol_NN                      = Sol_NN,
                    PDE_NN                      = PDE_NN,
                    Time_Derivative_Order       = Settings.PDE_Time_Derivative_Order,
                    Spatial_Derivative_Order    = Settings.PDE_Spatial_Derivative_Order,
                    Collocation_Coords          = Test_Colloc_Coords,
                    Data_Coords                 = Data_Container.Test_Data_Coords,
                    Data_Values                 = Data_Container.Test_Data_Values,
                    Data_Type                   = torch.float32,
                    Device                      = Settings.Device);

                (Train_Coll_Loss[i], Train_Data_Loss[i]) = Discovery_Testing(
                    Sol_NN                      = Sol_NN,
                    PDE_NN                      = PDE_NN,
                    Time_Derivative_Order       = Settings.PDE_Time_Derivative_Order,
                    Spatial_Derivative_Order    = Settings.PDE_Spatial_Derivative_Order,
                    Collocation_Coords          = Train_Colloc_Coords,
                    Data_Coords                 = Data_Container.Train_Data_Coords,
                    Data_Values                 = Data_Container.Train_Data_Values,
                    Data_Type                   = torch.float32,
                    Device                      = Settings.Device);

                # Print losses!
                print("Epoch #%-4d | Test: \t Collocation = %.7f\t Data = %.7f\t Total = %.7f"
                    % (t, Test_Coll_Loss[i], Test_Data_Loss[i], Test_Coll_Loss[i] + Test_Data_Loss[i]));
                print("            | Train:\t Collocation = %.7f\t Data = %.7f\t Total = %.7f"
                    % (Train_Coll_Loss[i], Train_Data_Loss[i], Train_Coll_Loss[i] + Train_Data_Loss[i]));

                # Increment the counter.
                Loss_Counter += 1;
            else:
                print(("Epoch #%-4d | "   % t));

    elif(Settings.Mode == "Extraction"):
        # Setup Extraction Coords
        Extraction_Coords = Generate_Random_Coords(
                Dim_Lower_Bounds    = Data_Container.Dim_Lower_Bounds,
                Dim_Upper_Bounds    = Data_Container.Dim_Upper_Bounds,
                Num_Points          = Settings.Num_Extraction_Points,
                Data_Type           = torch.float32,
                Device              = Settings.Device);

        # Generate the library!
        (PDE_NN_At_Coords,
         Library,
         num_multi_indices,
         multi_indices_list) = Generate_Library(
                                    Sol_NN                      = Sol_NN,
                                    PDE_NN                      = PDE_NN,
                                    Time_Derivative_Order       = Settings.PDE_Time_Derivative_Order,
                                    Spatial_Derivative_Order    = Settings.PDE_Spatial_Derivative_Order,
                                    Coords                      = Extraction_Coords,
                                    Poly_Degree                 = Settings.Extracted_Term_Degree,
                                    Device                      = Settings.Device);

        # Recursively find a sequence of candidate least squares solutions.
        (X, Residual) = Recursive_Feature_Elimination(
                            A = Library,
                            b = PDE_NN_At_Coords);

        # Rank the solutions according to the change in residual.
        (X_Ranked,
         Residual_Ranked,
         Residual_Change) = Rank_Candidate_Solutions(
                                X        = X,
                                Residual = Residual);

        # Pint the 5 most likely PDEs.
        Num_Cols : int = Library.shape[1];
        for i in range(Num_Cols - 5, Num_Cols):
            print(("The #%u most likely PDE gives a residual of %.4lf (%.2lf%% better than the next sparsest PDE)." % (Num_Cols - i, Residual_Ranked[i], Residual_Change[i]*100)));
            Print_Extracted_PDE(
                Extracted_PDE           = X_Ranked[:, i],
                Time_Derivative_Order   = Settings.PDE_Time_Derivative_Order,
                num_multi_indices       = num_multi_indices,
                multi_indices_list      = multi_indices_list);


    else:
        print(("Mode is %s. It should be one of \"PINNs\", \"Discovery\", \"Extraction\"." % Settings.Mode));
        print("Something went wrong. Aborting. Thrown by main.");
        exit();


    # Epochs are done. Figure out how long they took!
    Main_Time = Main_Timer.Stop();

    if (Settings.Mode == "PINNs" or Settings.Mode == "Discovery"):
        # In these modes, training can take hours. Thus, it's usually more
        # useful to report the time in minutes, seconds.
        Minutes : int   = int(Main_Time) // 60;
        Seconds : float = Main_Time - 60*Minutes;

        print("Running %d epochs took %um,%.2fs (%fs)." % (Epochs, Minutes, Seconds, Main_Time));
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
                    "Optimizer_State"   : Optimizer.state_dict()},
                    Save_File_Path);


if __name__ == '__main__':
    main();
