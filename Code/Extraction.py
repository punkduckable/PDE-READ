import numpy as np;
import torch;
from typing import Tuple;

from Network import Neural_Network;
from PDE_Residual import Evaluate_u_Derivatives;



def Recursive_Counter(
        n_sub_index_values  : int,
        degree              : int,
        sub_index           : int = 0,
        sub_index_value     : int = 0,
        counter             : int = 0) -> int:
    """ This function determines the number of "distinct" multi-indices of
    specified degree whose sub-indices take values in 0, 1... n_sub_index_values - 1.
    Here, two multi-indices are "the same" if and only if the indices in one
    of them can be rearranged into the other. This defines an equivalence
    relation of multi-indices. Thus, we are essentially finding the number of
    equivalence classes.

    For example, if n_sub_index_values = 4 and degree = 2, then the set of
    possible multi-indices is { (0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2),
    (1, 3), (2, 2), (2, 3), (3, 3) }, which contains 10 elements. Thus, in this
    case, this function would return 10.

    Note: we assume that degree and n_sub_index_values are POSITIVE integers.

    ----------------------------------------------------------------------------
    Arguments:

    n_sub_index_values: The number of distinct values that any one of the
    sub-indices can take on. If n_sub_index_values = k, then each sub index
    can take on values 0, 1,... k-1.

    degree: The number of sub-indices in the multi-index.

    sub_index: keeps track of which sub-index we're working on.

    sub_index_value: specifies which value we put in a particular sub-index.

    counter: the variable that actually stores the total number of
    multi-indices of specified oder whose sub-indices take values in 0, 1...
    n_sub_index_values - 1. This is what's eventually returned.

    ----------------------------------------------------------------------------
    Returns:

    The total number of "distinct" multi-indices (as defined above) of
    specified degree such that each sub-index takes values in 0, 1...
    n_sub_index_values - 1. """

    # Assertions.
    assert (degree > 0), \
        ("Degree must be a POSITIVE integer. Got %d." % degree);
    assert (n_sub_index_values > 0), \
        ("n_sub_index_values must be a POSITIVE integer. Got %d." % n_sub_index_values);

    # Base case
    if (sub_index == degree - 1):
        return counter + (n_sub_index_values - sub_index_value);

    # Recursive case.
    else : # if (sub_index < degree - 1):
        for j in range(sub_index_value, n_sub_index_values):
            counter = Recursive_Counter(
                        n_sub_index_values  = n_sub_index_values,
                        degree              = degree,
                        sub_index           = sub_index + 1,
                        sub_index_value     = j,
                        counter             = counter);

        return counter;



def Recursive_Multi_Indices(
        multi_indices       : np.array,
        n_sub_index_values  : int,
        degree              : int,
        sub_index           : int = 0,
        sub_index_value     : int = 0,
        position            : int = 0) -> int:
    """ This function essentially finds the set of "distinct" multi-indices of
    specified degree such that each sub-index in the multi-index takes values in
    0,1... n_sub_index_values-1. Here, two multi-indices are "the same" if and
    only if the indices in one of them can be rearranged into the other. This
    defines an equivalence relation of multi-indices. Thus, we are essentially
    finding a representative for each equivalence class under this relation.

    We assume that multi_indices is a N by degree array, where N is
    "sufficiently large" (meaning that N is at least as large as the value
    returned by Recursive_Counter with the n_sub_index_values and degree
    arguments). This function each row of multi_indices with a multi-index. The
    i, j element of multi_indices contains the value of the jth sub-index of the
    ith "distinct" (as defined above) multi-index.

    For example, if n_sub_index_values = 4 and degree = 2, then the set of
    "distinct" multi-indices is { (0, 0), (0, 1), (0, 2), (0, 3), (1, 1),
    (1, 2), (1, 3), (2, 2), (2, 3), (3, 3) }. This function will populate the
    first 10 rows of multi_indices with the following:
        [0, 0]
        [0, 1]
        [0, 2]
        [0, 3]
        [1, 1]
        [1, 2]
        [1, 3]
        [2, 2]
        [2, 3]
        [3, 3]

    Note: we assume that degree and n_sub_index_values are POSITIVE integers.

    ----------------------------------------------------------------------------
    Arguments:

    multi_indices: An "sufficiently large" array which will hold all distinct
    multi-indices of specified degree whose sub-indices take values in 0, 1,...
    n_sub_index_values - 1.

    n_sub_index_values: The number of distinct values that any sub-index can
    take on. If n_sub_index_values = k, then each sub_index can take values
    0, 1,... n_sub_index_values - 1.

    degree: The number of sub-indices in the multi-index.

    sub_index: keeps track of which sub-index we're working on.

    sub_index_value: specifies which value we put in a particular sub-index.

    ----------------------------------------------------------------------------
    Returns:

    an interger used for recursive purposes. You probably want to discard it. """

    # Assertions.
    assert (degree > 0), \
        ("Degree must be a POSITIVE integer. Got %d." % degree);
    assert (n_sub_index_values > 0), \
        ("n_sub_index_values must be a POSITIVE integer. Got %d." % n_sub_index_values);

    # Base case
    if (sub_index == degree - 1):
        for j in range(sub_index_value, n_sub_index_values):
            multi_indices[position + (j - sub_index_value), sub_index] = j;

        return position + (n_sub_index_values - sub_index_value);

    # Recursive case.
    else : # if (sub_index < degree - 1):
        for j in range(sub_index_value, n_sub_index_values):
            new_position = Recursive_Multi_Indices(
                            multi_indices       = multi_indices,
                            n_sub_index_values  = n_sub_index_values,
                            degree              = degree,
                            sub_index           = sub_index + 1,
                            sub_index_value     = j,
                            position            = position);

            for k in range(0, (new_position - position)):
                multi_indices[position + k, sub_index] = j;

            position = new_position;

        return position;



def Generate_Library(
        u_NN            : Neural_Network,
        N_NN            : Neural_Network,
        Coords          : torch.Tensor,
        num_derivatives : int,
        Poly_Degree     : int) -> np.array:
    """ This function populates the library matrix in the SINDY algorithm. How
    this works (and why it works the way that it does) is a tad involved.... so
    buckle up, here comes a (rather verbose) explanation:

    To populate the library, we need to know the set of all polynomial terms
    whose degree is <= Poly_Degree. To determine that set, we essentially need to
    find the set of all "distinct" multi-indices with up to Poly_Degree
    sub-indices. Here, two multi-indices are "equivalent" if the sub-indices in
    one multi-index can be rearranged to match that of the other multi-index
    (the same up to rearrangements of the sub-index ordering).

    Why does this give us what we want? Let's consider when Poly_Degree = 2 and
    num_derivatives = 2. In this case, our library consists of the following:
        degree 0: c,
        degree 1: u, du_dx, d^2u_dx^2
        degree 2: (u)^2, u*du_dx, u*d^2u_dx^2, (du_dx)^2, du_dx*d^2u_dx^2, (d^2u_dx^2)^2
    We want to reframe this as a multi-index problem. In particular, let us
    associate a multi-index to each term above. nth-degree terms will consist of
    multi-indices with n sub-indices, with one sub-index for each component of
    the polynomial term. Let us associate u with 0, du_dx with 1, and d^2u_dx^2
    with 2. Then, for example, the product u*du_dx will be associated with the
    multi-index (0, 1). Notice here that the product u*du_dx is the same as
    du_dx*u (because multiplication commutes). Only one of these should appear
    in the library. In other words, we want to treat the multi-index (0, 1) and
    (1, 0) as equivalent. However, this is precisely my definition of
    multi-index equivalence written above.

    Once we know the set of all "distinct" multi-indices, we can easily
    construct the library. In particular, we first allocate an array that has a
    column for each distinct multi-index with at most Poly_Degree sub-indices.
    We then evaluate u, du_dx,... at each point in Coords. If the ith column of
    the library corresponds to the multi-index (1, 3), then we pointwise
    multiply du_dx and du^3_dx^3 and store the product in that column.

    Note: This function only works if u depends on one spatial variable.

    ----------------------------------------------------------------------------
    Arguments:

    u_NN: The network that approximates the PDE solution.

    N_NN: Neural network that approximates the PDE.

    Coords: The coordinates of the extraction points (The library will contain
    a row for each element of Coords).

    num_derivatives: The highest order spatial derivaitive that will be included
    in the library (this is also the highest order derivative that we think is
    in the governing PDE)

    Poly_Degree: the maximum degree of the polynomial terms in the library. For
    example, if we expect to extract a linear PDE, then Poly_Degree should be 1.
    Setting Poly_Degree > 1 allows the algorithm to search for non-linear PDEs.

    ----------------------------------------------------------------------------
    Returns:

    A 4 element tuple. For the first, we evaluate u, u_xx,... at each point in
    Coords. We then evaluate N_NN at each of these collection of values. This
    is the first element of the returned tuple. The second holds the library
    (with a row for each coordinate and a column for each term in the library).
    The third holds a Poly_Degree element array whose ith element holds the
    number of multi_indices of degree k whose sub-indices take values in {0,
    1,... num_derivatives}. The 4th is a list of arrays, the ith one of which
    holds the set of multi-indices of order i whose sub-indices take values in
    {0, 1,... num_derivatives}. """

    # We need a sub-index value for each derivative, as well as u itself.
    n_sub_index_values = num_derivatives + 1;

    # Determine how many multi-indices with k sub-indices exist for each
    # k in {0, 1,... Poly_Degree}.
    num_multi_indices    = np.empty(Poly_Degree + 1, dtype = np.int);
    num_multi_indices[0] = 1;
    for i in range(1, Poly_Degree + 1):
        num_multi_indices[i] = Recursive_Counter(
                                    n_sub_index_values  = n_sub_index_values,
                                    degree              = i);

    # Set up a list to hold the multi-index arrays of each degree.
    multi_indices_list = [];
    multi_indices_list.append(np.array(1, dtype = np.int));

    # Use this information to initialize the Library as a tensor of ones.
    # We need everything to be ones because of how we populate this matrix (see
    # below).
    num_rows : int = Coords.shape[0];
    num_cols : int = num_multi_indices.sum();
    Library : np.array = np.ones((num_rows, num_cols), dtype = np.float32);

    # Evaluate u, du/dx,... at each point.
    (du_dt, diu_dxi) = Evaluate_u_Derivatives(
                            u_NN            = u_NN,
                            num_derivatives = num_derivatives,
                            Coords          = Coords);

    # Evaluate n at the output given by diu_dxi.
    N_NN_batch = N_NN(diu_dxi);

    # Now populate the library the multi-index approach described above. Note
    # that the first column corresponds to a constant and should, therefore,
    # be filled with 1's (which it already is).
    position = 1;
    for degree in range(1, Poly_Degree + 1):
        # Create a buffer to hold the multi-indices of this degree.
        multi_indices = np.empty((num_multi_indices[degree], degree), dtype = np.int);

        # Find the set of multi indices for this degree!
        Recursive_Multi_Indices(
            multi_indices       = multi_indices,
            n_sub_index_values  = n_sub_index_values,
            degree              = degree);

        # Cycle through the multi-indices of this degree
        for i in range(num_multi_indices[degree]):

            # Cycle through the sub-indices of this multi-index.
            for j in range(degree):
                Library[:, position] = (Library[:, position]*
                                        diu_dxi[:, multi_indices[i, j]].detach().squeeze().numpy());

            # Increment position
            position += 1;

        # Append this set of multi_indices for this degree to the list.
        multi_indices_list.append(multi_indices);

    # All done, the library is now populated! Package everything together and
    # return.
    return (N_NN_batch.detach().squeeze().numpy(),
            Library,
            num_multi_indices,
            multi_indices_list);



def Thresholded_Least_Squares(
        A         : np.array,
        b         : np.array,
        threshold : float) -> np.array:
    """ This problem solves a thresholded leat squares problem. That is, it
    essentially finds arg_min(||Ax - b||_{2}). The big difference is that we try
    to eliminate the component of x which are smaller than the threshold. In
    particular, we first find x(1) = arg_min(||Ax - b||_{2}). We then find set
    to zero all components of x(1) whose absolute value is less than the
    threshold. We eliminate the corresponding columns of A to get A(1) (which
    now only has those columns of A for which the corresponding component of x
    is at least as big as the threshold). We then solve x(2) =
    arg_min(||A(1)x - b||_{2}) and repeat.

    ----------------------------------------------------------------------------
    Arguments:

    A, b: The matrix A and vector b in Ax = b.

    threshold: the smallest value we allow components of x to take on. Any
    component of x whose absolute value is less than the threshold will be
    zeroed out.

    ----------------------------------------------------------------------------
    Returns:

    A numpy array which is basically arg_min(Ax - b). See description above.
    If A has m columns, then the returned vector should have m components. """

    # Solve the initial least squares problem.
    x : np.array = np.linalg.lstsq(A, b, rcond = None)[0];

    # Perform the thresholding procedure.
    for k in range(0, 5):
        # Determine which components of x are smaller than the threshold. This
        # yields a boolean vector, whose ith component of x is smaller than
        # the threshold, and 0 otherwise.
        small_indices : np.array = (abs(x) < threshold);
        x[small_indices] = 0;
        print("Eliminated %d components after step %d of thresholded least squares." % (small_indices.sum(), k));

        # Now determine which components of x are bigger than the threshold.
        big_indices : np.array   = np.logical_not(small_indices);

        # Resolve least squares problem but only using the columns of A
        # corresponding to the big columns.
        x[big_indices] = np.linalg.lstsq(A[:, big_indices], b, rcond = None)[0];

    # All done, return x!
    return x;



def Print_Extracted_PDE(
        Extracted_PDE      : np.array,
        num_multi_indices  : np.array,
        multi_indices_list : Tuple) -> None:
    """ This function takes in the output of the Thresholded_Least_Squares
    function and turns it into a PDE in a human readable format.

    Note: this function only works if u (the solution) is a function of 1
    spatial variable.

    ----------------------------------------------------------------------------
    Arguments:

    Extracted_PDE: This should be the thresholded least squares solution (
    returned by Thresholded_Least_Squares).

    num_multi_indices, multi_indices_list: the corresponding variables returned
    by Generate_Library.

    ----------------------------------------------------------------------------
    Returns:

    Nothing! """


    # Start the printout.
    print("Extracted the following PDE:");
    print("du/dt =", end = '');

    if (Extracted_PDE[0] != 0):
        print("%f " % Extracted_PDE[0], end = '');

    position : int    = 1;
    for degree in range(1, num_multi_indices.shape[0]):
        # Cycle through the multi-indices of this degree.
        for i in range(num_multi_indices[degree]):

            # If this term of the extracted PDE is non-zero, print out the
            # corresponding term.
            if(Extracted_PDE[position] != 0):
                print(" + %f" % Extracted_PDE[position], end = '');
                multi_index = multi_indices_list[degree][i];

                # cycle through the sub-indices of this multi-index.
                for j in range(degree):
                    if(multi_index[j] == 0):
                        print("(u)", end = '');
                    else:
                        print("(d^%d u/dx^%d)" % (multi_index[j], multi_index[j]), end = '');
            position += 1;

    # Finish printing (this just pints a new line character).
    print();
