import numpy as np;
import torch;



class Neural_Network(torch.nn.Module):
    def __init__(self,
                 Num_Hidden_Layers : int = 3,
                 Neurons_Per_Layer : int = 20,   # Neurons in each Hidden Layer
                 Input_Dim : int         = 1,    # Dimension of the input
                 Output_Dim : int        = 1):   # Dimension of the output
        # Note: we assume that Num_Hidden_Layers, Neurons_Per_Layer, Input_Dim,
        # and out_dim are positive integers.
        assert (Num_Hidden_Layers > 0   and
                Neurons_Per_Layer > 0   and
                Input_Dim > 0           and
                Output_Dim > 0), "Neural_Network initialization arguments must be positive integers!"

        # Call the superclass initializer.
        super(Neural_Network, self).__init__();

        # Define object attributes. Note that there is an optput layer in
        # addition to the hidden layers (which is why Num_Layers is
        # Num_Hidden_Layers + 1)
        self.Input_Dim  : int = Input_Dim;
        self.Output_Dim : int = Output_Dim;
        self.Num_Layers : int = Num_Hidden_Layers + 1;

        # Define Layers ModuleList.
        self.Layers = torch.nn.ModuleList();

        # Append the first hidden layer. The domain of this layer is the input
        # domain, which means that in_features = Input_Dim. Since this is a
        # hidden layer, however it must have Neurons_Per_Layer output features.
        self.Layers.append(
            torch.nn.Linear(    in_features  = Input_Dim,
                                out_features = Neurons_Per_Layer,
                                bias = True )
        );

        # Now append the rest of the hidden layers. Each of these layers maps
        # from \mathbb{R}^{Neurons_Per_Layer} to itself. Thus, in_features =
        # out_features = Neurons_Per_Layer. We start at i = 1 because we already
        # setup the 1st hidden layer.
        for i in range(1, Num_Hidden_Layers):
            self.Layers.append(
                torch.nn.Linear(    in_features  = Neurons_Per_Layer,
                                    out_features = Neurons_Per_Layer,
                                    bias = True )
            );

        # Now, append the Output Layer, which has Neurons_Per_Layer input
        # features, but only Output_Dim output features.
        self.Layers.append(
            torch.nn.Linear(    in_features  = Neurons_Per_Layer,
                                out_features = Output_Dim,
                                bias = True )
        );

        # Initialize the weight matricies, bias vectors in the network.
        for i in range(self.Num_Layers):
            torch.nn.init.xavier_uniform_(self.Layers[i].weight);

        # Finally, set the Network's activation function.
        self.Activation_Function = torch.nn.Tanh();



    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """ Forward method for the NN class (to enable calling the network).
        Note that the user should not call this function directly. Rather,
        they should call it through the __call__ method (using the NN object
        like a function), which is defined in the module class and calls
        forward.

        ------------------------------------------------------------------------
        Arguments:
        x : A batch of inputs. This should be a B by Input_Dim tensor, whose
        ith row holds the ith input (this is how the Linear function works).
        """

        # Pass x through the network's layers!
        for i in range(self.Num_Layers - 1):
            x = self.Activation_Function(self.Layers[i](x));

        # Pass through the last layer and return (there is no activation
        # function in the last layer)
        return self.Layers[self.Num_Layers - 1](x);
