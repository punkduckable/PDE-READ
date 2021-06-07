# Nonsense to add Code diectory to the python search path.
import os
import sys

# Get path to parent directory
parent_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)));

# Add the Code directory to the python path.
Code_path   = os.path.join(parent_dir, "Code");
sys.path.append(Code_path);

# Now we can do our usual import stuff.
import numpy as np;
import torch;
import unittest;
import Network;
import Loss_Functions;


class Test_Network(unittest.TestCase):
    def test_Network(self):
        Neurons_Per_Hidden_Layer : int = 2;

        # Make a simple Neural Network.
        NN : Neural_Network = Network.Neural_Network(
                                    Num_Hidden_Layers = 1,
                                    Neurons_Per_Layer = Neurons_Per_Hidden_Layer,
                                    Input_Dim         = 2,
                                    Output_Dim        = 1);

        # Confirm it has two layers (a single hidden layer and the output layer)
        self.assertEqual(len(NN.Layers), 2);

        # Confirm the hidden layer has the correct number of Neurons. This
        # is the number of rows of the weight matrix of the first layer.
        self.assertEqual(NN.Layers[0].weight.data.shape[0], Neurons_Per_Hidden_Layer);

        # Manually set the Network's weights, biases.
        torch.nn.init.ones_(NN.Layers[0].weight.data);
        torch.nn.init.ones_(NN.Layers[1].weight.data);
        torch.nn.init.ones_(NN.Layers[0].bias.data);
        torch.nn.init.ones_(NN.Layers[1].bias.data);

        # The network now has a predictable form. In particular,
        # NN(x) = 2tanh(x[0] + x[1] + 1) + 1
        # (think about it). To test generality, we will let x be random.
        x = torch.rand((1, 2), dtype = torch.float32);

        NN_x_actual  = NN(x);
        NN_x_predict = 2.0*torch.tanh(x[0,0] + x[0,1] + 1.0) + 1.0;

        # Check that actual and predicted outputs are sufficiently close.
        self.assertLess(abs(NN_x_actual - NN_x_predict).item(), .00001);



class Test_Loss_Functions(unittest.TestCase):
    def test_IC_Loss(self):
        # First, make a simple network.
        u_NN : Neural_Network = Network.Neural_Network(
                                    Num_Hidden_Layers = 1,
                                    Neurons_Per_Layer = 2,
                                    Input_Dim         = 2,
                                    Output_Dim        = 1);

        # Manually set the network Weights and Biases.
        torch.nn.init.ones_(u_NN.Layers[0].weight.data);
        torch.nn.init.ones_(u_NN.Layers[1].weight.data);


"""
# I stole this from Extraction.py... turn it into a unit test!
def main():
    # Initialize parameters.
    n_sub_index_values  = 4;
    degree              = 3;

    # Run Recursive_Counter to determine how big x must be.
    counter = Recursive_Counter(
                n_sub_index_values  = n_sub_index_values,
                degree              = degree);

    # allocate space for x.
    multi_indices = np.empty((counter, degree), dtype = np.int);

    # Populate x using Recursive_Multi_Indices
    Recursive_Multi_Indices(
        multi_indices       = multi_indices,
        n_sub_index_values  = n_sub_index_values,
        degree              = degree);



    # Print results.
    print(counter);
    print(multi_indices);
"""


if(__name__ == "__main__"):
    unittest.main();
