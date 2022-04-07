import numpy;
import torch;
import matplotlib.pyplot as plt;

# Nonsense to add Code diectory to the python search path.
import os;
import sys;

# Get path to parent directory
parent_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)));

# Add the Code directory to the python path.
Code_path   = os.path.join(parent_dir, "Code");
sys.path.append(Code_path);

from Network        import Neural_Network;
from PDE_Residual   import Evaluate_Derivatives;



def Plot_Static_Beam() -> None:
    # Generate some coordinates.
    x_low       : float         = 0.0;
    x_high      : float         = 20.0;
    Coords      : numpy.ndarray = numpy.linspace(x_low, x_high, num = 500, dtype = numpy.float32);
    T_Coords    : torch.Tensor  = torch.from_numpy(Coords).view(-1, 1);

    # Load the Solution network.
    Sol_NN = Neural_Network( Num_Hidden_Layers   = 3,
                             Neurons_Per_Layer   = 10,
                             Input_Dim           = 1,
                             Output_Dim          = 1,
                             Activation_Function = "Rational",
                             Batch_Norm          = False);

    Load_File_Path : str = "../Saves/Static_Beam_LBFGS"
    Saved_State = torch.load(Load_File_Path, map_location=torch.device('cpu'));
    Sol_NN.load_state_dict(Saved_State["Sol_Network_State"]);

    # Evaluate the network at the coordinates.
    Sol_At_Coords : numpy.ndarray = Sol_NN(T_Coords).view(-1).detach().numpy();

    # Evaluate Sol_NN's second derivative at the coordinates.
    Dxn_U = Evaluate_Derivatives(
                        Sol_NN                      = Sol_NN,
                        Time_Derivative_Order       = 1,
                        Spatial_Derivative_Order    = 2,
                        Coords                      = T_Coords,
                        Data_Type                   = torch.float32,
                        Device                      = torch.device('cpu'));

    Dx2_U_At_Coords : numpy.ndarray = Dxn_U[:, 2].view(-1).detach().numpy();

    plt.plot(Coords, Sol_At_Coords);
    plt.plot(Coords, Dx2_U_At_Coords);
    plt.show();



if __name__ == "__main__":
    Plot_Static_Beam();
