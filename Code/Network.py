import numpy as np;
import torch;



class Rational_ReLU(torch.nn.Module):
    def __init__(self, Data_Type = torch.float32):
        # Based on the following paper (see appendix A for coefficients):
        # Boulle, Nicolas, Yuji Nakatsukasa, and Alex Townsend. "Rational neural
        # networks." arXiv preprint arXiv:2004.01902 (2020).

        super(Rational_ReLU, self).__init__();

        # Initialize numerator and denominator coefficients to the best
        # rational function approximation to ReLU.
        self.a = torch.nn.parameter.Parameter(torch.tensor((1.1915, 1.5957, 0.5, .0218), dtype = Data_Type));
        self.a.requires_grad_(True);

        self.b = torch.nn.parameter.Parameter(torch.tensor((2.3830, 0.0, 1.0), dtype = Data_Type));
        self.b.requires_grad_(True);

    def forward(self, X : torch.tensor):
        """This method elementwise applies the rational function to X.

        ------------------------------------------------------------------------
        Arguments:

        X : A B by n tensor, where B is the batch size, and n is the number of
        neurons in some layer of some neural network.

        ------------------------------------------------------------------------
        Returns:

        Let N(x) = sum_{i = 0}^{3} a_i x^i and D(x) = sum_{i = 0}^{2} b_i x^i.
        Let R = N/D (ignoring points where D(x) = 0). This function applies R
        to each element of X and returns the resulting tensor. """

        # Create aliases for self.a and self.b so to make the code cleaner
        a = self.a;
        b = self.b;

        # Evaluate the numerator and denominator.
        N_X = a[0] + X*(a[1] + X*(a[2] + a[3]*X));
        D_X = b[0] + X*(b[1] + b[2]*X);

        # Return R = N_X/D_X. This also evalutes elementwise.
        return N_X/D_X;



class Neural_Network(torch.nn.Module):
    def __init__(self,
                 Num_Hidden_Layers   : int         = 3,
                 Neurons_Per_Layer   : int         = 20,   # Neurons in each Hidden Layer
                 Input_Dim           : int         = 1,    # Dimension of the input
                 Output_Dim          : int         = 1,    # Dimension of the output
                 Data_Type           : torch.dtype = torch.float32,
                 Activation_Function : str         = "Tanh"):
        # Note: Fo the code below to work, Num_Hidden_Layers, Neurons_Per_Layer,
        # Input_Dim, and out_dim must be positive integers.
        assert (Num_Hidden_Layers > 0   and
                Neurons_Per_Layer > 0   and
                Input_Dim         > 0   and
                Output_Dim        > 0), \
                "Neural_Network initialization arguments must be positive integers!"

        # Call the superclass initializer.
        super(Neural_Network, self).__init__();

        # Define object attributes. Note that there is an optput layer in
        # addition to the hidden layers (which is why Num_Layers is
        # Num_Hidden_Layers + 1)
        self.Input_Dim  : int = Input_Dim;
        self.Output_Dim : int = Output_Dim;
        self.Num_Layers : int = Num_Hidden_Layers + 1;

        # Define Layers ModuleList.
        self.Layers               = torch.nn.ModuleList();

        # Append the first hidden layer. The domain of this layer is the input
        # domain, which means that in_features = Input_Dim. Since this is a
        # hidden layer, however it must have Neurons_Per_Layer output features.
        self.Layers.append(
            torch.nn.Linear(    in_features  = Input_Dim,
                                out_features = Neurons_Per_Layer,
                                bias = True ).to(dtype = Data_Type));


        # Now append the rest of the hidden layers. Each of these layers maps
        # from \mathbb{R}^{Neurons_Per_Layer} to itself. Thus, in_features =
        # out_features = Neurons_Per_Layer. We start at i = 1 because we already
        # setup the 1st hidden layer.
        for i in range(1, Num_Hidden_Layers):
            self.Layers.append(
                torch.nn.Linear(    in_features  = Neurons_Per_Layer,
                                    out_features = Neurons_Per_Layer,
                                    bias = True ).to(dtype = Data_Type));

        # Now, append the Output Layer, which has Neurons_Per_Layer input
        # features, but only Output_Dim output features.
        self.Layers.append(
            torch.nn.Linear(    in_features  = Neurons_Per_Layer,
                                out_features = Output_Dim,
                                bias = True ).to(dtype = Data_Type));

        # Initialize the weight matricies, bias vectors in the network.
        for i in range(self.Num_Layers):
            torch.nn.init.xavier_uniform_(self.Layers[i].weight);

        # Finally, set the Network's activation functions.
        self.Activation_Functions = torch.nn.ModuleList();
        if  (Activation_Function == "Tanh"):
            for i in range(Num_Hidden_Layers):
                self.Activation_Functions.append(torch.nn.Tanh());
        elif(Activation_Function == "Sigmoid"):
            for i in range(Num_Hidden_Layers):
                self.Activation_Functions.append(torch.nn.Sigmoid());
        elif(Activation_Function == "Rational"):
            for i in range(Num_Hidden_Layers):
                self.Activation_Functions.append(Rational_ReLU());
        else:
            print("Unknown Activation Function. Got %s" % Activation_Function);
            print("Thrown by Neural_Network.__init__. Aborting.");
            exit();



    def forward(self, X : torch.Tensor) -> torch.Tensor:
        """ Forward method for the NN class (to enable calling the network).
        Note that the user should not call this function directly. Rather,
        they should call it through the __call__ method (using the NN object
        like a function), which is defined in the module class and calls
        forward.

        ------------------------------------------------------------------------
        Arguments:

        x: A batch of inputs. This should be a B by Input_Dim tensor, whose
        ith row holds the ith input (this is how the Linear function works).

        ------------------------------------------------------------------------
        Returns:

        A tensor containing the value of the network ealuated at X. """

        # Pass X through the network's layers!
        for i in range(self.Num_Layers - 1):
            X = self.Activation_Functions[i](self.Layers[i](X));

        # Pass through the last layer and return (there is no activation
        # function in the last layer)
        return self.Layers[self.Num_Layers - 1](X);
