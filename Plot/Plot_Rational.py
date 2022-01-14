# Nonsense to add Code diectory to the python search path.
import os
import sys

# Get path to parent directory
parent_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)));

# Add the Code directory to the python path.
Code_path   = os.path.join(parent_dir, "Code");
sys.path.append(Code_path);

import numpy;
import torch;
from   matplotlib import cm;
import matplotlib.pyplot as plt;

import Network;


def Generate_Complex_Grid(
        Re_Low        : float,
        Re_High       : float,
        Im_Low        : float,
        Im_High       : float,
        Num_Re_Values : int,
        Num_Im_Values : int):
    """ This function is used to generate a uniformly spaced grid of complex
    values in the rectangle defined by Re_Low, Re_High, Im_Low, and Im_High. """

    # declare a torch tensor to hold the complex numbers.
    Complex_Grid = torch.empty((Num_Re_Values, Num_Im_Values), dtype = torch.complex64);
    Real_Grid = Complex_Grid.real;
    Imag_Grid = Complex_Grid.imag;

    # Find the set of possible Real, Imasginary values.
    Real_Values = numpy.linspace(Re_Low, Re_High, num = Num_Re_Values);
    Imag_Values = numpy.linspace(Re_Low, Re_High, num = Num_Re_Values);

    # Use these to populate the real and imaginary parts of the grid points.
    for i in range(Num_Re_Values):
        for j in range(Num_Im_Values):
            Real_Grid[i, j] = Real_Values[i];
            Imag_Grid[i, j] = Imag_Values[j];

    # All done!
    return Complex_Grid;


if(__name__ == "__main__"):
    # Set bounds
    Re_Low  : float = -8;
    Re_High : float =  8;
    Im_Low  : float = -8;
    Im_High : float =  8;

    # Set number of grid points
    Num_Re_Values : int = 500;
    Num_Im_Values : int = 500;

    Complex_Grid = Generate_Complex_Grid(
                        Re_Low  = Re_Low,
                        Re_High = Re_High,
                        Im_Low  = Im_Low,
                        Im_High = Im_High,
                        Num_Re_Values = Num_Re_Values,
                        Num_Im_Values = Num_Im_Values);

    # Loaded network Architecture.
    Num_Hidden_Layers : int = 5;
    Neurons_Per_Layer : int = 50;

    # Set up a network. Note that we use complex64 types.
    Sol_NN = Network.Neural_Network(
                        Num_Hidden_Layers = Num_Hidden_Layers,
                        Neurons_Per_Layer = Neurons_Per_Layer,
                        Input_Dim         = 2,
                        Output_Dim        = 1,
                        Data_Type         = torch.complex64,
                        Activation_Function = "Rational");

    # Load in saved network.
    Saved_State = torch.load("../Saves/Burgers_Adam_Rational_100");
    Sol_NN.load_state_dict(Saved_State["Sol_Network_State"]);

    # Pick time to evaluate solution at.
    t : float = 4.0;

    # Now, construct input coordinates.
    Position_Coords = Complex_Grid.view(-1, 1);
    Time_Coords     = torch.full_like(Position_Coords, t);
    Coords          = torch.hstack((Time_Coords, Position_Coords));

    # Evaluate Sol_NN at each coordinate. We reshape this to look the same as
    # the Complex_Grid.
    Sol_At_Coords = Sol_NN(Coords).view(Complex_Grid.size());

    # Get magnitude, angle information from solution data.
    Sol_Mag   = Sol_At_Coords.abs();
    Sol_Angle = Sol_At_Coords.angle();

    # Cap maginutde of output (otherwise, it kinda looks like junk).
    Num_Rows : int = Complex_Grid.size()[0];
    Num_Cols : int = Complex_Grid.size()[1];
    for i in range(Num_Rows):
        for j in range(Num_Cols):
            if(Sol_Mag[i, j] > 5.0):
                Sol_Mag[i,j ] = 5.0;

    # Get real, imaginary parts of the coordinates.
    Real_Values = Complex_Grid.real;
    Imag_Values = Complex_Grid.imag;

    # Convert everything to numpy arrays
    Sol_Mag_np   = Sol_Mag.detach().numpy();
    Sol_Angle_np = Sol_Angle.detach().numpy();
    Real_Values_np = Real_Values.numpy();
    Imag_Values_np = Imag_Values.numpy();

    # Make the plot.
    Fig = plt.figure(figsize = (10, 5));
    Ax_Mag   = Fig.add_subplot(1, 2, 1);
    Ax_Angle = Fig.add_subplot(1, 2, 2);

    Ax_Mag.contourf  (Real_Values_np, Imag_Values_np, Sol_Mag_np,   100, cmap = cm.jet);
    Ax_Angle.contourf(Real_Values_np, Imag_Values_np, Sol_Angle_np, 100, cmap = cm.jet);

    Ax_Mag.set_xlabel("Re(z)");
    Ax_Mag.set_ylabel("Im(z)");
    Ax_Mag.set_title("Magnitude");

    Ax_Angle.set_xlabel("Re(z)");
    Ax_Angle.set_ylabel("Im(z)");
    Ax_Angle.set_title("Angle");

    Fig.tight_layout();
    plt.show();
