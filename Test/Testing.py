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
import random;

import Network;
import Loss_Functions;
import PDE_Residual;
import Extraction;



# Global machine epsilon variable (for comparing floating point results)
Epsilon : float = .000005;


def One_Initialize_Network(Hidden_Neurons : int) -> Network.Neural_Network:
    """ This function initializes a single-hidden-layer neural network with a
    variable number of neurons in the first layer. Every weight and bias in the
    network is initialized to a tensor of ones. This gives the network a
    predictable (though still nonlinear) output, which is very useful for
    testing. In particular, the network we create should evaluate as follows:
        u_NN(x) = n*tanh(x[0] + x[1] + 1) + 1
    where n = Hidden_Neurons. """

    # Initialize the network.
    NN = Network.Neural_Network(
                Num_Hidden_Layers = 1,
                Neurons_Per_Layer = Hidden_Neurons,
                Input_Dim         = 2,
                Output_Dim        = 1);

    # Manually set the network Weights and Biases to tensors of ones.
    torch.nn.init.ones_(NN.Layers[0].weight.data);
    torch.nn.init.ones_(NN.Layers[1].weight.data);
    torch.nn.init.ones_(NN.Layers[0].bias.data);
    torch.nn.init.ones_(NN.Layers[1].bias.data);

    return NN;



class Test_Network(unittest.TestCase):
    def test_Network(self):
        # Make a simple Neural Network.
        Hidden_Neurons : int = random.randint(1, 50);
        NN : Network.Neural_Network = One_Initialize_Network(Hidden_Neurons);

        # Confirm it has 2 layers (a single hidden layer and the output layer)
        self.assertEqual(len(NN.Layers), 2);

        # Confirm the hidden layer has the correct number of Neurons. This
        # is the number of rows of the weight matrix of the first layer.
        self.assertEqual(NN.Layers[0].weight.data.shape[0], Hidden_Neurons);

        # The network now has a predictable form. In particular,
        # NN(x) = n*tanh(x[0] + x[1] + 1) + 1
        # where n = Hidden_Neurons. (think about it). To test generality, we
        # will let x be random.
        num_Points : int = random.randint(1, 1000);
        x = torch.rand((num_Points, 2), dtype = torch.float32);

        NN_x_Actual  = NN(x);
        NN_x_Predict = Hidden_Neurons*torch.tanh(x[:, 0] + x[:, 1] + 1.0) + 1.0;

        # Check that actual and predicted outputs are sufficiently close.
        for i in range(num_Points):
            self.assertLess(abs(NN_x_Actual[i] - NN_x_Predict[i]).item(), Epsilon);



class Test_Loss_Functions(unittest.TestCase):
    def test_IC_Loss(self):
        # First, initialize a simple network.
        Hidden_Neurons : int          = random.randint(1, 100);
        u_NN : Network.Neural_Network = One_Initialize_Network(Hidden_Neurons);

        # Make up some random initial condition coordinates and data.
        num_IC_Points : int = random.randint(10, 1000);
        IC_Coords           = torch.rand((num_IC_Points, 2), dtype = torch.float32);
        IC_Data             = torch.rand(num_IC_Points, dtype = torch.float32);

        # We expect the IC error to take the following form:
        #       IC_Error = (1/n) sum_{i = 0}^n (u_NN(IC_Coords[i]) - IC_Data[i])^2
        # (where n = num_IC_Points).
        # In theory, the network should evaluate to the following:
        #       u_NN(x) = m*tanh(x[0] + x[1] + 1) + 1
        # where m = Hidden_Neurons.

        # Calculate predicted IC error.
        IC_Error_Sum = torch.tensor(0, dtype = torch.float32);
        for i in range(num_IC_Points):
            u_NN_pt = Hidden_Neurons*torch.tanh(IC_Coords[i, 0] + IC_Coords[i, 1] + 1.0) + 1.0;
            IC_Error_Sum += (u_NN_pt - IC_Data[i])**2;

        IC_Error_Predict = IC_Error_Sum/num_IC_Points;

        # Now compute the actual IC loss.
        IC_Error_Actual = Loss_Functions.IC_Loss(
                                u_NN      = u_NN,
                                IC_Coords = IC_Coords,
                                IC_Data   = IC_Data);

        # Check that pediction is "sufficiently close" to actual.
        self.assertLess(abs(IC_Error_Actual - IC_Error_Predict).item(), num_IC_Points*Epsilon);



    def test_BC_Loss(self):
        # First, make a simple network.
        Hidden_Neurons : int          = random.randint(1, 100);
        u_NN : Network.Neural_Network = One_Initialize_Network(Hidden_Neurons);

        # The network should now evaluate as follows:
        #   u_NN(x) = n*tanh(x[0] + x[1] + 1) + 1
        # where n = Hidden_Neurons. Thus,
        #   (d/dx[i])u_NN(x) = n*tanh'(x[0] + x[1] + 1)
        #                    = n*(1 - tanh^2(x[0] + x[1] + 1))
        # for i = 0, 1.

        # Make up some random Boundary coordinates.
        Num_BC_Points : int = random.randint(10, 1000);
        Lower_Bound_Coords  = torch.rand((Num_BC_Points, 2), dtype = torch.float32);
        Upper_Bound_Coords  = torch.rand((Num_BC_Points, 2), dtype = torch.float32);

        # Evaluate u and u' at the left and right boundary. Use these to
        # construct the predicted BC loss.
        U_Sq_Er   = torch.tensor(0, dtype = torch.float32);
        U_x_Sq_Er = torch.tensor(0, dtype = torch.float32);

        for i in range(Num_BC_Points):
            u_low    = Hidden_Neurons*torch.tanh(Lower_Bound_Coords[i, 0] + Lower_Bound_Coords[i, 1] + 1.0) + 1.0;
            u_high   = Hidden_Neurons*torch.tanh(Upper_Bound_Coords[i, 0] + Upper_Bound_Coords[i, 1] + 1.0) + 1.0;
            U_Sq_Er += (u_low - u_high)**2;

            u_x_low    = Hidden_Neurons*(1 - torch.tanh(Lower_Bound_Coords[i, 0] + Lower_Bound_Coords[i, 1] + 1.0)**2);
            u_x_high   = Hidden_Neurons*(1 - torch.tanh(Upper_Bound_Coords[i, 0] + Upper_Bound_Coords[i, 1] + 1.0)**2);
            U_x_Sq_Er += (u_x_low - u_x_high)**2;

        BC_Loss_Predict = U_Sq_Er/Num_BC_Points + U_x_Sq_Er/Num_BC_Points;

        # Now evaluate the actual BC loss.
        BC_Loss_Actual  = Loss_Functions.Periodic_BC_Loss(
                                u_NN               = u_NN,
                                Lower_Bound_Coords = Lower_Bound_Coords,
                                Upper_Bound_Coords = Upper_Bound_Coords,
                                Highest_Order      = 1);

        # Check that pediction is "sufficiently close" to actual.
        self.assertLess(abs(BC_Loss_Predict - BC_Loss_Actual).item(), Num_BC_Points*Epsilon);



    # Note: We don't test Collocation Loss since this loss function literally
    # just computes the mean square PDE_Residual.... it's only a few lines long.
    # We have another test for the PDE_Residual. See below.



    def test_Data_Loss(self):
        # First, set up a simple network.
        Hidden_Neurons : int          = random.randint(1, 100);
        u_NN : Network.Neural_Network = One_Initialize_Network(Hidden_Neurons);

        # Make up some random data points, values. .
        Num_Data_Points : int = random.randint(10, 1000);
        Data_Coords           = torch.rand((Num_Data_Points, 2), dtype = torch.float32);
        Data_Values           = torch.rand(Num_Data_Points, dtype = torch.float32);

        # Calculuate predicted data Loss.
        Data_Error = torch.tensor(0, dtype = torch.float32);
        for i in range(Num_Data_Points):
            u_pt        = Hidden_Neurons*torch.tanh(Data_Coords[i, 0] + Data_Coords[i, 1] + 1) + 1;
            Data_Error += (u_pt - Data_Values[i])**2;

        Data_Loss_Predict = Data_Error/Num_Data_Points;

        # Now compute actual Data Loss.
        Data_loss_Actual = Loss_Functions.Data_Loss(
                                u_NN        = u_NN,
                                Data_Coords = Data_Coords,
                                Data_Values = Data_Values);

        # Check that pediction is "sufficiently close" to actual.
        self.assertLess(abs(Data_Loss_Predict - Data_loss_Actual).item(), Num_Data_Points*Epsilon);



class Test_PDE_Residual(unittest.TestCase):
    def test_Evaluate_u_Derivatives(self):
        # Set up a simple network.
        Hidden_Neurons : int          = random.randint(1, 100);
        u_NN : Network.Neural_Network = One_Initialize_Network(Hidden_Neurons);

        # The network should now evaluate as follows:
        #   u_NN(t, x) = n*tanh(t + x + 1) + 1
        # where n = Hidden_Neurons. Thus,
        #   (d/dz)u_NN(t, x) = n*tanh'(t + x + 1)
        #                    = n*(1 - tanh^2(t + x + 1))
        # for z = t, x. And thus,
        #   (d^2/dx^2)u_NN(t, x) = -2*n*tanh(t + x + 1)*tanh'(t + x+ 1)
        #                        = -2*n*tanh(t + x + 1)*[1 - tanh^2(t + x + 1)]

        # Set up some random points to evaluate u and its derivatives.
        num_Points = random.randint(10, 1000);
        Coords     = torch.rand((num_Points, 2), dtype = torch.float32);

        # Compute predicted value for du_dt, du_dx, and d^2u_dx^2.
        u_predict       = torch.empty(num_Points, dtype = torch.float32);
        du_dt_predict   = torch.empty(num_Points, dtype = torch.float32);
        du_dx_predict   = torch.empty(num_Points, dtype = torch.float32);
        d2u_dx2_predict = torch.empty(num_Points, dtype = torch.float32);
        for i in range(num_Points):
            t = Coords[i, 0];
            x = Coords[i, 1];
            n = Hidden_Neurons;

            u_predict[i]        = n*torch.tanh(t + x+  1.0) + 1.0;
            du_dt_predict[i]   = n*(1.0 - torch.tanh(t + x + 1.0)**2);
            du_dx_predict[i]   = n*(1.0 - torch.tanh(t + x + 1.0)**2);
            d2u_dx2_predict[i] = -2.0*n*torch.tanh(t + x + 1.0)*(1.0 - torch.tanh(t + x + 1.0)**2);

        # Now compute actual du_dt, du_dx, d2u_dx2.
        (du_dt_actual, diu_dxi_actual) = PDE_Residual.Evaluate_u_Derivatives(
                                                u_NN            = u_NN,
                                                num_derivatives = 2,
                                                Coords          = Coords);
        u_actual       = diu_dxi_actual[:, 0];
        du_dx_actual   = diu_dxi_actual[:, 1];
        d2u_dx2_actual = diu_dxi_actual[:, 2];

        # Compare actual, predicted values!
        for i in range(num_Points):
            self.assertLess(abs(u_predict[i]       - u_actual[i]      ).item(), 2.0*Epsilon);
            self.assertLess(abs(du_dt_predict[i]   - du_dt_actual[i]  ).item(), 5.0*Epsilon);
            self.assertLess(abs(du_dx_predict[i]   - du_dx_actual[i]  ).item(), 5.0*Epsilon);
            self.assertLess(abs(d2u_dx2_predict[i] - d2u_dx2_actual[i]).item(), 9.0*Epsilon);



    # Note: We do not test PDE_Residual, since this function basically
    # just calls Evaluate_u_Derivatives and then pushes one of the returned
    # values through N_NN and compares the output to the other returned value.
    # All of these operations are already covered/tested elsewhere, so it
    # doesn't really make sense to write a test for this function as well.
    # This is the same reason why I don't have a test for Collocation_Loss.



class Test_Extraction(unittest.TestCase):
    def test_Recursive_Counter(self):
        # Test the Recursive Counter function with a few inputs for which we
        # know the correct answer.

        # If we allow one sub-index, which takes values in in {1, 2,... n} then
        # there are possible multi-indices.
        n : int             = random.randint(1, 100);
        num_indices_predict = n;
        num_indices_actual  = Extraction.Recursive_Counter(
                                    num_sub_index_values = n,
                                    degree               = 1);
        self.assertEqual(num_indices_predict, num_indices_actual);



        # If we allow 2 sub-indices, each of which can take values in {1, 2,...
        # n}, then there are (n+1)(n/2) possible multi-indices. They are
        # the following:
        #  (1, 1)
        #  (1, 2) (2, 2)
        #  (1, 3) (2, 3) (3, 3)
        #   ...    ...    ...
        #  (1, n) (2, n) (3, n) ... (n, n)
        n : int = random.randint(1, 100);

        num_indices_predict = ((n + 1)*n)//2;
        num_indices_actual  = Extraction.Recursive_Counter(
                                    num_sub_index_values = n,
                                    degree               = 2);

        self.assertEqual(num_indices_predict, num_indices_actual);



        # If we allow 3 sub-indices, each of which can take values in {1, 2,...
        # n}, then there are sum_{k = 1}^{n} (k + 1)(k/2) possible multi-indices
        # They are the following:
        # (1,1,1)                     (2,2,2)
        # (1,1,2) (1,2,2)             (2,2,3) (2,3,3)
        #   ...     ...                 ...     ...                   (n-1,n-1,n-1)
        # (1,1,n) (1,2,n) ... (1,n,n) (2,2,n) (2,3,n) ... (2,n,n) ... (n-1,n-1,n  ) (n-1,n,n) (n,n,n)
        n : int             = random.randint(1, 100);
        num_indices_predict = 0;
        for k in range(1, n+1):
            num_indices_predict += ((k + 1)*k)//2;

        num_indices_actual = Extraction.Recursive_Counter(
                                num_sub_index_values = n,
                                degree               = 3);

        self.assertEqual(num_indices_predict, num_indices_actual);



    def test_Recursive_Multi_Indices(self):
        # If we allow 1 sub-index which takes on values in {1, 2,... n} then
        # the possible multi-indices are 0, 1, 2,... n-1.
        n : int     = random.randint(1, 1000);
        num_indices = Extraction.Recursive_Counter(
                                num_sub_index_values = n,
                                degree               = 1);
        multi_indices = np.empty((num_indices, 1), dtype = np.float);
        Extraction.Recursive_Multi_Indices(
                        multi_indices        = multi_indices,
                        num_sub_index_values = n,
                        degree               = 1);

        # Check that each possible sub index is in Multi-indices.
        multi_indices_list = multi_indices.tolist();
        for k in range(n):
            self.assertIn([k], multi_indices_list);



        # If we allow 2 sub-indices which take values in {1, 2,... n} then the
        # possible multi-indices are the following:
        #  (1, 1)
        #  (1, 2) (2, 2)
        #  (1, 3) (2, 3) (3, 3)
        #   ...    ...    ...
        #  (1, n) (2, n) (3, n) ... (n, n)
        n : int     = random.randint(1, 100);
        num_indices = Extraction.Recursive_Counter(
                                num_sub_index_values = n,
                                degree               = 2);
        multi_indices = np.empty((num_indices, 2), dtype = np.float);
        Extraction.Recursive_Multi_Indices(
                        multi_indices        = multi_indices,
                        num_sub_index_values = n,
                        degree               = 2);

        multi_indices_list = multi_indices.tolist();
        for i in range(n):
            for j in range(i, n):
                self.assertIn([i, j], multi_indices_list);



        # If we allow 3 sub-indices which take values in {1, 2,... n} then the
        # possible multi-indices are the following:
        # (1,1,1)                     (2,2,2)
        # (1,1,2) (1,2,2)             (2,2,3) (2,3,3)
        #   ...     ...                 ...     ...                   (n-1,n-1,n-1)
        # (1,1,n) (1,2,n) ... (1,n,n) (2,2,n) (2,3,n) ... (2,n,n) ... (n-1,n-1,n  ) (n-1,n,n) (n,n,n)
        n : int     = random.randint(1, 10);
        num_indices = Extraction.Recursive_Counter(
                                num_sub_index_values = n,
                                degree               = 3);
        multi_indices = np.empty((num_indices, 3), dtype = np.float);
        Extraction.Recursive_Multi_Indices(
                        multi_indices        = multi_indices,
                        num_sub_index_values = n,
                        degree               = 3);

        multi_indices_list = multi_indices.tolist();
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    self.assertIn([i, j, k], multi_indices_list);



    def test_Generate_Library(self):
        # Make two simple neural networks. One for u and one for N. The N
        # network will (by virtue of how One_Initialize_Network works) be a
        # function of u and du/dx.
        u_Hidden_Neurons : int        = random.randint(1, 10);
        u_NN : Network.Neural_Network = One_Initialize_Network(u_Hidden_Neurons);

        N_Hidden_Neurons : int        = random.randint(1, 10);
        N_NN : Network.Neural_Network = One_Initialize_Network(N_Hidden_Neurons);

        # u_NN and N_NN should take the following form:
        #   u_NN(x, t )  = n_u*tanh(x + t  + 1) + 1
        #   N_NN(u, u_x) = n_N*tanh(u + u' + 1) + 1
        # where n_u = u_Hidden_Neurons, n_N = N_Hidden_Neurons, and u_x = du/dx.
        # Further,
        #   u_x(x, t) = n_u*(1 - tanh^2(x + t + 1))

        # Generate some random points to evaluate the network at.
        num_Coords : int = random.randint(20, 40);
        Coords           = torch.rand((num_Coords, 2), dtype = torch.float32);

        # Evaluate u_NN (and its derivatives) at each coordinate.
        u_at_Coords   = np.empty(num_Coords, dtype = np.float32);
        u_x_at_Coords = np.empty(num_Coords, dtype = np.float32);
        for i in range(num_Coords):
            n = u_Hidden_Neurons;
            t = Coords[i, 0];
            x = Coords[i, 1];

            u_at_Coords[i]   = n*torch.tanh(t + x + 1.0) + 1.0;
            u_x_at_Coords[i] = n*(1 - torch.tanh(t + x + 1)**2);

        # Now generate a library. We will allow terms of degree <= 2. There
        # should be 6 such terms.
        Library_Predict       = np.empty((num_Coords, 6), dtype = np.float32);
        Library_Predict[:, 0] = 1;                         # const
        Library_Predict[:, 1] = u_at_Coords;               # u
        Library_Predict[:, 2] = u_x_at_Coords;             # du/dx
        Library_Predict[:, 3] = u_at_Coords**2;            # u*u
        Library_Predict[:, 4] = u_at_Coords*u_x_at_Coords; # u*du/dx
        Library_Predict[:, 5] = u_x_at_Coords**2;          # du/dx*du/dx

        # Evaluate the actual library and compare.
        (N_Coords,
         Library_Actual,
         num_multi_indices,
         multi_indices_list) = Extraction.Generate_Library(
                                        u_NN            = u_NN,
                                        N_NN            = N_NN,
                                        Coords          = Coords,
                                        num_derivatives = 1,
                                        Poly_Degree     = 2);

        self.assertEqual(Library_Actual.shape, Library_Predict.shape);
        for i in range(num_Coords):
            for j in range(6):
                self.assertLess(abs(Library_Actual[i, j] - Library_Predict[i, j]).item(), 10*Epsilon);



if(__name__ == "__main__"):
    unittest.main();
