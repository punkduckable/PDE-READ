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



# Global machine epsilon variable (for comparing floating point results)
Epsilon : float = .00002;

class Test_Network(unittest.TestCase):
    def test_Network(self):
        Neurons_Per_Hidden_Layer : int = 2;

        # Make a simple Neural Network.
        NN : Network.Neural_Network = Network.Neural_Network(
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

        NN_x_Actual  = NN(x);
        NN_x_Predict = 2.0*torch.tanh(x[0,0] + x[0,1] + 1.0) + 1.0;

        # Check that actual and predicted outputs are sufficiently close.
        self.assertLess(abs(NN_x_Actual - NN_x_Predict).item(), Epsilon);



def One_Initialize_Network(Hidden_Neurons : int) -> Network.Neural_Network:
    """ This function initializes a single-hidden-layer neural network with a
    variable number of neurons in the first layer. Every weight and bias in the
    network is initialized to a tensor of ones. This gives the network a
    predictable (though still nonlinear) output, which is very useful for
    testing. In particular, the network we create should evaluate as follows:
        u_NN(x) = n*tanh(x[0] + x[1] + 1) + 1
    where n = Hidden_Neurons. """

    # Initialize the network.
    u_NN = Network.Neural_Network(
                Num_Hidden_Layers = 1,
                Neurons_Per_Layer = Hidden_Neurons,
                Input_Dim         = 2,
                Output_Dim        = 1);

    # Manually set the network Weights and Biases to tensors of ones.
    torch.nn.init.ones_(u_NN.Layers[0].weight.data);
    torch.nn.init.ones_(u_NN.Layers[1].weight.data);
    torch.nn.init.ones_(u_NN.Layers[0].bias.data);
    torch.nn.init.ones_(u_NN.Layers[1].bias.data);

    return u_NN;




class Test_Loss_Functions(unittest.TestCase):
    def test_IC_Loss(self):
        # First, initialize a simple network.
        Hidden_Neurons : int = 8;
        u_NN = One_Initialize_Network(Hidden_Neurons);

        # Make up some random initial condition coordinates and data.
        num_IC_Points = 50;
        IC_Coords = torch.rand((num_IC_Points, 2), dtype = torch.float32);
        IC_Data   = torch.rand(num_IC_Points, dtype = torch.float32);

        # We expect the IC error to take the following form:
        #       IC_Error = (1/n) sum_{i = 0}^n (u_NN(IC_Coords[i]) - IC_Data[i])^2
        # (where n = num_IC_Points).
        # In theory, the network should evaluate to the following:
        #       u_NN(x) = 2*tanh(x[0] + x[1] + 1) + 1
        IC_Error_Sum = torch.tensor(0, dtype = torch.float32);
        for i in range(num_IC_Points):
            u_NN_pt = Hidden_Neurons*torch.tanh(IC_Coords[i, 0] + IC_Coords[i, 1] + 1) + 1;
            IC_Error_Sum += (u_NN_pt - IC_Data[i])**2;

        IC_Error_Predict = IC_Error_Sum/num_IC_Points;

        # Now compute the actual IC loss.
        IC_Error_Actual = Loss_Functions.IC_Loss(
                                u_NN      = u_NN,
                                IC_Coords = IC_Coords,
                                IC_Data   = IC_Data);

        # Check that pediction is "sufficiently close" to actual.
        self.assertLess(abs(IC_Error_Actual - IC_Error_Predict).item(), Epsilon);



    def test_BC_Loss(self):
        # First, make a simple network.
        Hidden_Neurons : int = 5;
        u_NN = One_Initialize_Network(Hidden_Neurons);

        # The network should now evaluate as follows:
        #   u_NN(x) = n*tanh(x[0] + x[1] + 1) + 1
        # where n = Hidden_Neurons. Thus,
        #   (d/dx[i])u_NN(x) = n*tanh'(x[0] + x[1] + 1)
        #                    = n*(1 - tanh^2(x[0] + x[1] + 1))
        # for i = 0, 1.

        # Make up some random Boundary coordinates.
        Num_BC_Points : int = 30;
        Lower_Bound_Coords = torch.rand((Num_BC_Points, 2), dtype = torch.float32);
        Upper_Bound_Coords = torch.rand((Num_BC_Points, 2), dtype = torch.float32);

        # Evaluate u and u' at the left and right boundary. Use these to
        # construct the predicted BC loss.
        U_Sq_Er   = torch.tensor(0, dtype = torch.float32);
        U_x_Sq_Er = torch.tensor(0, dtype = torch.float32);

        for i in range(Num_BC_Points):
            u_low    = Hidden_Neurons*torch.tanh(Lower_Bound_Coords[i, 0] + Lower_Bound_Coords[i, 1] + 1) + 1;
            u_high   = Hidden_Neurons*torch.tanh(Upper_Bound_Coords[i, 0] + Upper_Bound_Coords[i, 1] + 1) + 1;
            U_Sq_Er += (u_low - u_high)**2;

            u_x_low    = Hidden_Neurons*(1 - torch.tanh(Lower_Bound_Coords[i, 0] + Lower_Bound_Coords[i, 1] + 1)**2);
            u_x_high   = Hidden_Neurons*(1 - torch.tanh(Upper_Bound_Coords[i, 0] + Upper_Bound_Coords[i, 1] + 1)**2);
            U_x_Sq_Er += (u_x_low - u_x_high)**2;

        BC_Loss_Predict = U_Sq_Er/Num_BC_Points + U_x_Sq_Er/Num_BC_Points;

        # Now evaluate the actual BC loss.
        BC_Loss_Actual  = Loss_Functions.Periodic_BC_Loss(
                                u_NN               = u_NN,
                                Lower_Bound_Coords = Lower_Bound_Coords,
                                Upper_Bound_Coords = Upper_Bound_Coords,
                                Highest_Order      = 1);

        # Check that pediction is "sufficiently close" to actual.
        self.assertLess(abs(BC_Loss_Predict - BC_Loss_Actual).item(), Epsilon);



    # Note: We don't test Collocation Loss since this loss function literally
    # just computes the mean square PDE_Residual.... it's only a few lines long.
    # We have another test for the PDE_Residual. See below.



    def test_Data_Loss(self):
        # First, set up a simple network.
        Hidden_Neurons : int = 7;
        u_NN : Network.Neural_Network = One_Initialize_Network(Hidden_Neurons);

        # Make up some random data points, values. .
        Num_Data_Points = 40;
        Data_Coords = torch.rand((Num_Data_Points, 2), dtype = torch.float32);
        Data_Values = torch.rand(Num_Data_Points, dtype = torch.float32);

        # Calculuate predicted data Loss.
        Data_Error = torch.tensor(0, dtype = torch.float32);
        for i in range(Num_Data_Points):
            u_pt = Hidden_Neurons*torch.tanh(Data_Coords[i, 0] + Data_Coords[i, 1] + 1) + 1;
            Data_Error += (u_pt - Data_Values[i])**2;

        Data_Loss_Predict = Data_Error/Num_Data_Points;

        # Now compute actual Data Loss.
        Data_loss_Actual = Loss_Functions.Data_Loss(
                                u_NN = u_NN,
                                Data_Coords = Data_Coords,
                                Data_Values = Data_Values);

        # Check that pediction is "sufficiently close" to actual.
        self.assertLess(abs(Data_Loss_Predict - Data_loss_Actual).item(), Epsilon);



class Test_PDE_Residual(unittest.TestCase):
    def test_Evaluate_u_Derivatives(self):
        # To do!
        pass;

    def Test_PDE_Residual(self):
        # To do!
        pass;



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
