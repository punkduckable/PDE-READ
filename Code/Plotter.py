import matplotlib.pyplot as plt;
import torch;
import numpy as np;
from typing import Tuple;

from Network import Neural_Network, PDE_Residual;



# Determine how well the network satisifies the PDE at a set of point.
def Evaluate_Residuals(u_NN : Neural_Network, N_NN : Neural_Network, Coords : torch.Tensor) -> np.array:
    """ For brevity, let u = u_NN and N = N_NN. At each coordinate, this
    function computes
                    du/dt - N(u, du_dx,... )
    which we call the residual. If the NN perfectly satisified the PDE, then the
    residual would be zero everywhere. However, since the NN only approximates
    the PDE solution, we get non-zero residuals.

    ----------------------------------------------------------------------------
    Arguments:
    u_NN : the Neural Network that approximates solution to the learned PDE.

    N_NN : The Neural Network that approximates the PDE.

    Coords : a tensor of coordinates of points where we want to evaluate the
    residual This must be a N by 2 tensor, where N is the number of points. The
    ith row of this tensor should contain the x,t coordinates of the ith point
    where we want to evaluate the residual.

    ----------------------------------------------------------------------------
    Returns:
    a numpy array. The ith element of this array gives the residual at the ith
    element of points. """

    # First, determine the number of points and intiailize the residual array.
    num_points : int = Coords.shape[0];
    Residual = np.empty((num_points), dtype = np.float);

    for i in range(num_points):
        # Get the xy coordinate of the ith collocation point.
        xt = Coords[i];

        # Evaluate the Residual from the learned PDE at this point.
        Residual[i] = PDE_Residual(u_NN, N_NN, xt).item();

    return Residual;



# Evaluate solution at a set of points.
def Evaluate_Approx_Sol(u_NN : Neural_Network, Coords : torch.Tensor) -> np.array:
    """ This function evaluates the approximate solution at each element of
    Coords.

    ----------------------------------------------------------------------------
    Arguments:
    u_NN : the Neural Network that approximates solution to the learned PDE.

    Coords : The set of points where we want to evaluate the approximate
    solution. This should be a Nx2 tensor of floats whose ith row holds the x,t
    coordinates of the ith point where we want to evaluate the approximate
    solution.

    ----------------------------------------------------------------------------
    Returns:
    A numpy array whose ith element is the value of u_NN at the ith element of
    Coords. If Coords is a Nx2 tensor, then this is a N element numpy array. """

    # Get number of points, initialize the u array.
    num_Points : int = Coords.shape[0];
    u_NN_at_Points = np.empty((num_Points), dtype = np.float);

    # Loop through the points, evaluate the network at each one.
    for i in range(num_Points):
        u_NN_at_Points[i] = u_NN.forward(Coords[i]).item();

    return u_NN_at_Points;



# Set up Axes objects for plotting
def Setup_Axes() -> Tuple[plt.figure, np.array]:
    """ This function sets up the figure, axes objects for plotting. There
    are a lot of settings to tweak, so I thought the code would be cleaner
    if those details were outsourced to this function.

    ----------------------------------------------------------------------------
    Arguments:
    None!

    ----------------------------------------------------------------------------
    Returns:
    A tuple. The first element contains the figure object, the second contains
    a numpy array of axes objects (to be passed to Update_Axes). """

    # Set up the figure object.
    fig = plt.figure(figsize = (12, 4));

    # Approx solution subplot.
    Axes1 = fig.add_subplot(1, 3, 1);
    Axes1.set_title("Neural Network Approximation");

    # True solution subplot
    Axes2 = fig.add_subplot(1, 3, 2);
    Axes2.set_title("True Solution");

    # Residual subplot.
    Axes3 = fig.add_subplot(1, 3, 3);
    Axes3.set_title("PDE Residual");

    # Package axes objects into an array.
    Axes = np.array([Axes1, Axes2, Axes3]);

    # Set settings that are the same for each Axes object.
    # The domain of each Axes object is the unit (2) square, and it's aspect
    # ratio should be equal. I set these parameters in a loop so that I only
    # have to type them once, thereby improving code maintainability.
    for i in range(3):
        # Set x, y bounds
        Axes[i].set_xbound(0., 1.);
        Axes[i].set_ybound(0., 1.);

        # Force python to produce a square plot.
        Axes[i].set_aspect('equal', adjustable = 'datalim');
        Axes[i].set_box_aspect(1.);

    return (fig, Axes);



# The plotting function!
def Update_Axes(fig : plt.figure, Axes : np.ndarray, u_NN : Neural_Network, N_NN : Neural_Network, Coords : torch.Tensor, True_Sol_at_Points : np.array) -> None:
    """ This function plots the approximate solution and residual at the
    specified points.

    ----------------------------------------------------------------------------
    Arguments:
    fig : The figure object to which the Axes belong. We need this to set up
    the color bars.

    Axes : The array of Axes object that we will plot on. Note that this
    function will overwrite these axes.

    u_NN : the Neural Network that approximates solution to the learned PDE.

    N_NN : The Neural Network that approximates the PDE.

    Coords : The coordinates of the points we want to evaluate the approximate
    and true solutions, as well as the PDE Residual.

    True_Sol_at_Points : A numpy array containing the true solution at each
    element of Coords. 

    ----------------------------------------------------------------------------
    Returns:
    Nothing! """

    # First, evaluate the network's approximate solution, the true solution, and
    # the PDE residual at the specified Points. We need to reshape these into
    # nxn grids, because that's what matplotlib's contour function wants. It's
    # annoying, but it is what it is.
    u_NN_at_Points      = Evaluate_Approx_Sol(u_NN, Coords).reshape(n,n);
    Residual_at_Points  = Evaluate_Residuals(u_NN, N_NN, Coords).reshape(n,n);

    # Extract the x and y coordinates of points, as np arrays. We also need to
    # reshape these as nxn grids (same reason as above.
    x = Points[:, 0].numpy().reshape(n,n);
    y = Points[:, 1].numpy().reshape(n,n);

    # Plot the approximate solution + colorbar.
    ColorMap0 = Axes[0].contourf(x, y, u_NN_at_Points.reshape(n,n), levels = 50, cmap = plt.cm.jet);
    fig.colorbar(ColorMap0, ax = Axes[0], fraction=0.046, pad=0.04, orientation='vertical');

    # Plot the true solution + colorbar
    ColorMap1 = Axes[1].contourf(x, y, True_Sol_at_Points.reshape(n,n), levels = 50, cmap = plt.cm.jet);
    fig.colorbar(ColorMap1, ax = Axes[1], fraction=0.046, pad=0.04, orientation='vertical');

    # Plot the residual + colorbar
    ColorMap2 = Axes[2].contourf(x, y, Residual_at_Points.reshape(n,n), levels = 50, cmap = plt.cm.jet);
    fig.colorbar(ColorMap2, ax = Axes[2], fraction=0.046, pad=0.04, orientation='vertical');

    # Set tight layout (to prevent overlapping... I have no idea why this isn't
    # a default setting. Matplotlib, you are special kind of awful).
    fig.tight_layout();
