# Nonsense to add Code diectory to the python search path.
import os
import sys

# Get path to parent directory
parent_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)));

# Add the Code directory to the python path.
Code_path   = os.path.join(parent_dir, "Code");
sys.path.append(Code_path);

# Now we can import code files.
import Network;
import Timing;

import numpy;
import torch;



def Profile_Activation_Functions():
    # First, initialize two networks. One will use Rational activation functions
    # and the other will use Tanh.
    Rational_NN = Network.Neural_Network(
                            Num_Hidden_Layers   = 5,
                            Neurons_Per_Layer   = 100,
                            Input_Dim           = 3,
                            Output_Dim          = 1,
                            Activation_Function = "Rational");
    Tanh_NN = Network.Neural_Network(
                            Num_Hidden_Layers   = 5,
                            Neurons_Per_Layer   = 100,
                            Input_Dim           = 3,
                            Output_Dim          = 1,
                            Activation_Function = "Tanh");

    # Now generate some random data.
    Data = torch.rand(  (20000, 3),
                        dtype = torch.float32,
                        requires_grad = True);


    # Pass the data through both networks, time it.
    Rational_Timer = Timing.Timer();
    Tanh_Timer     = Timing.Timer();

    Rational_Timer.Start();
    Rational_NN(Data);
    Rational_Time = Rational_Timer.Stop();

    Tanh_Timer.Start();
    Tanh_NN(Data);
    Tanh_Time = Tanh_Timer.Stop();

    # Print results.
    print("Rational: %fs" % Rational_Time);
    print("Tanh:     %fs" % Tanh_Time);



if (__name__ == "__main__"):
    Profile_Activation_Functions();
