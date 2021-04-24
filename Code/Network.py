import torch;
import numpy as np;



class Neural_Network(torch.nn.Module):
    def __init__(self,
                 Num_Hidden_Layers : int = 3,    # Number of Hidden Layers
                 Nodes_Per_Layer : int   = 20,   # Nodes in each Hidden Layer
                 Input_Dim : int         = 1,    # Number of components in the input
                 Output_Dim : int        = 1):   # Number of components in the output
        # Note: we assume that Num_Hidden_Layers, Nodes_Per_Layer, Input_Dim,
        # and out_dim are positive integers.
        assert (Num_Hidden_Layers > 0   and
                Nodes_Per_Layer > 0     and
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
        # hidden layer, however it must have Nodes_Per_Layer output features.
        self.Layers.append(
            torch.nn.Linear(    in_features  = Input_Dim,
                                out_features = Nodes_Per_Layer,
                                bias = True )
        );

        # Now append the rest of the hidden layers. Each of these layers maps
        # within the same space, which means that in_features = out_features.
        # Note that we start at i = 1 because we already made the 1st
        # hidden layer.
        for i in range(1, Num_Hidden_Layers):
            self.Layers.append(
                torch.nn.Linear(    in_features  = Nodes_Per_Layer,
                                    out_features = Nodes_Per_Layer,
                                    bias = True )
            );

        # Now, append the Output Layer, which has Nodes_Per_Layer input
        # features, but only Output_Dim output features.
        self.Layers.append(
            torch.nn.Linear(    in_features  = Nodes_Per_Layer,
                                out_features = Output_Dim,
                                bias = True )
        );

        # Initialize the weight matricies, bias vectors in the network.
        for i in range(self.Num_Layers):
            torch.nn.init.xavier_uniform_(self.Layers[i].weight);

        # Finally, set the Network's activation function.
        self.Activation_Function = torch.nn.Tanh();

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # Note: the input must be an Input_Dim dimensional (1d) tensor.

        # Pass x through the network's layers!
        for i in range(self.Num_Layers - 1):
            x = self.Activation_Function(self.Layers[i](x));

        # Pass through the last layer and return (there is no activation
        # function in the last layer)
        return self.Layers[self.Num_Layers - 1](x);




# Loss from enforcing the PDE at the collocation points.
def Collocation_Loss(u_NN : Neural_Network, N_NN : Neural_Network, Collocation_Coords : torch.Tensor) -> torch.Tensor:
    """ This function evaluates how well u_NN satisifies the learned PDE at the
    collocation points. For brevity, let u = u_NN and N = N_NN. At each
    collocation point, we compute the following:
                                du/dt + N(u, du/dx, d^2u/dx^2)
    If u actually satisified the learned PDE, then this whould be zero everywhere.
    However, it generally won't be. This function computes the square of the
    quantity above at each Collocation point. We return the mean of these squared
    errors.

    ----------------------------------------------------------------------------
    Arguments:
    u_NN : The neural network that approximates the solution.

    N_NN : The neural network that approximates the PDE.

    Collocation_Coords : a tensor of coordinates of the collocation points. If
    there are N collocation points, then this should be a N x 2 tensor, whose
    ith row holds the x, t coordinate of the ith collocation point.

    ----------------------------------------------------------------------------
    Returns:
    Mean Square Error of the learned PDE at the collocation points. """

    num_Collocation_Points : int = Collocation_Coords.shape[0];

    # Now, initialize the loss and loop through the collocation points!
    Loss = torch.tensor(0, dtype = torch.float);
    for i in range(num_Collocation_Points):
        xt = Collocation_Coords[i];

        # We need to compute the gradeint of u with respect to the x t
        # coordinates.
        xt.requires_grad_(True);

        # Calculate approximate solution at this collocation point.
        u = u_NN(xt);

        # Compute gradient of u with respect to xt. We have to create the graph
        # used to compute grad_u so that we can evaluate second derivatives.
        # We also need to set retain_graph to True (which is implicitly set by
        # setting create_graph = True, though I keep it to make the code more
        # explicit) so that torch keeps the computational graph for u, which we
        # will need when we do backpropigation.
        grad_u = torch.autograd.grad(u, xt, retain_graph = True, create_graph = True)[0];

        # compute du/dx and du/dt. grad_u is a two element tensor. It's first
        # element holds du/dx, and its second element holds du/dt.
        du_dx = grad_u[0];
        du_dt = grad_u[1];

        # Now compute the gradients of du_dx with respect to xt. We
        # need to create graphs for this so that torch can track this operation
        # when constructing the computational graph for the loss function
        # (which it will use in backpropigation). We also need to retain
        # grad_u's graph for when we do backpropigation.
        grad_du_dx = torch.autograd.grad(du_dx, xt, retain_graph = True, create_graph = True)[0];

        # We want d^2u/dx^2, which should be the [0] element of grad_du_dx.
        d2u_dx2 = grad_du_dx[0];

        # Evaluate the learned operator N at this point.
        N_u_ux_uxx = N_NN(u, u_x, u_xx);

        # Evaluate the Learned PDE at this point.
        Loss += (du_dt - N_u_ux_uxx) ** 2;

    # Divide the accmulated loss by the number of collocation points to get
    # the mean square collocation loss.
    return (Loss / num_Collocation_Points);



# Loss from requiring the learned solution to satisify the training data.
def Data_Loss(u_NN : Neural_Network, Data_Coords : torch.Tensor, Data_Values : torch.Tensor) -> torch.Tensor:
    """ This function evaluates how well the learned solution u satisifies the
    training data. Specifically, for each point ((x_i, t_i), u_i) in
    data, we compute the square of the difference between u_i (the true
    solution at the point (x_i, t_i)) and u(x_i, t_i), where u denotes the
    learned solution. We return the mean of these squared errors. Here the
    phrase "data point" means "a point in the domain at which we know the value
    of the true solution"

    ----------------------------------------------------------------------------
    Arguments:
    u_NN : The neural network that approximates the solution.

    Data_Coords : A tensor of coordinates of the data points. If there are
    N data points, then this should be a N x 2 tensor, whose ith row holds the
    x, t coordinates of the ith data point.

    Data_Values : A tensor containing the value of the true solution at each
    data point. If there are N data points, then this should be an N element
    tesnor whose ith element holds the value of the true solution at the ith
    data point.

    ----------------------------------------------------------------------------
    Returns:
    Mean Square Error between the learned solution and the true solution at
    the data points. """

    num_Data_Points : int = Data_Coordinates.shape[0];

    # Now, initialize the Loss and loop through the Boundary Points.
    Loss = torch.tensor(0, dtype = torch.float);
    for i in range(num_Data_Points):
        xt = Boundary_Points[i];

        # Compute learned solution at this Data point.
        u_approx = u_NN(xt);

        # Get exact solution at this data point.
        u_true = Data_Values[i];

        # Aggregate square of difference between the required BC and the learned
        # solution at this data point.
        Loss += (u_appox - u_true)**2;

    # Divide the accmulated loss by the number of boundary points to get
    # the mean square boundary loss.
    return (Loss / num_Data_Points);
