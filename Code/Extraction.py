import numpy as np;
import torch;

from Network import Neural_Network;
from PDE_Residual import Evaluate_u_derivatives;



def Recursive_Counter(
        n_sub_index_values  : int,
        order               : int,
        sub_index           : int = 0,
        sub_index_value     : int = 0,
        counter             : int = 0) -> int:
    """ This function determines the number of "distinct" multi-indices of
    specified order whose sub-indices take values in 0, 1... n_sub_index_values - 1.
    Here, two multi-indices are "the same" if and only if the indices in one
    of them can be rearranged into the other. This defines an equivalence
    relation of multi-indices. Thus, we are essentially finding the number of
    equivalence classes.

    For example, if n_sub_index_values = 4 and order = 2, then the set of possible
    multi-indices is { (0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3),
    (2, 2), (2, 3), (3, 3) }, which contains 10 elements. Thus, in this case,
    this function would return 10.

    Note: we assume that order and n_sub_index_values are POSITIVE integers.

    ----------------------------------------------------------------------------
    Arguments:

    n_sub_index_values: The number of distinct values that any one of the
    sub-indicies can take on. If n_sub_index_values = k, then each sub index
    can take on values 0, 1,... k-1.

    order: The number of sub-indicies in the multi-index.

    sub_index: keeps track of which sub-index we're working on.

    sub_index_value: specifies which value we put in a particular sub-index.

    counter: the variable that actually stores the total number of
    multi-indices of specified oder whose sub-indices take values in 0, 1...
    n_sub_index_values - 1. This is what's eventually returned.

    ----------------------------------------------------------------------------
    Returns:

    The total number of "distinct" multi-indices (as defined above) of
    specified order such that each sub-index takes values in 0, 1...
    n_sub_index_values - 1. """

    # Assertions.
    assert (order > 0), \
        ("Order must be a POSITIVE integer. Got %d." % order);
    assert (n_sub_index_values > 0), \
        ("n_sub_index_values must be a POSITIVE integer. Got %d." % n_sub_index_values);

    # Base case
    if (sub_index == order - 1):
        return counter + (n_sub_index_values - sub_index_value);

    # Recursive case.
    else : # if (sub_index < order - 1):
        for j in range(sub_index_value, n_sub_index_values):
            counter = Recursive_Counter(
                        n_sub_index_values  = n_sub_index_values,
                        order               = order,
                        sub_index           = sub_index + 1,
                        sub_index_value     = j,
                        counter             = counter);

        return counter;



def Recursive_Multi_Indices(
        multi_indices       : np.array,
        n_sub_index_values  : int,
        order               : int,
        sub_index           : int = 0,
        sub_index_value     : int = 0,
        position            : int = 0) -> int:
    """ This function essentially finds the set of "distinct" multi-indices of
    specified order such that each sub-index in the multi-index takes values in
    0,1... n_sub_index_values-1. Here, two multi-indices are "the same" if and only
    if the indices in one of them can be rearranged into the other. This
    defines an equivalence relation of multi-indices. Thus, we are essentially
    finding a representative for each equivalence class under this relation.

    We assume that multi_indices is a N by order array, where N is
    "sufficiently large" (meaning that N is at least as large as the value
    returned by Recursive_Counter with the n_sub_index_values and order arguments).
    This function each row of multi_indices with a multi-index. The i, j
    element of multi_indices contains the value of the jth sub-index of the ith
    "distinct" (as defined above) multi-index.

    For example, if n_sub_index_values = 4 and order = 2, then the set of "distinct"
    multi-indices is { (0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3),
    (2, 2), (2, 3), (3, 3) }. This function will populate the first 10 rows of
    multi_indices with the following:
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

    Note: we assume that order and n_sub_index_values are POSITIVE integers.

    ----------------------------------------------------------------------------
    Arguments:

    multi_indices: An "sufficiently large" array which will hold all distinct
    multi-indices of specified order whose sub-indices take values in 0, 1,...
    n_sub_index_values - 1.

    n_sub_index_values: The number of distinct values that any sub-index can
    take on. If n_sub_index_values = k, then each sub_index can take values
    0, 1,... n_sub_index_values - 1.

    order: The number of sub-indicies in the multi-index.

    sub_index: keeps track of which sub-index we're working on.

    sub_index_value: specifies which value we put in a particular sub-index.

    ----------------------------------------------------------------------------
    Returns:

    an interger used for recursive purposes. You probably want to discard it. """

    # Assertions.
    assert (order > 0), \
        ("Order must be a POSITIVE integer. Got %d." % order);
    assert (n_sub_index_values > 0), \
        ("n_sub_index_values must be a POSITIVE integer. Got %d." % n_sub_index_values);

    # Base case
    if (sub_index == order - 1):
        for j in range(sub_index_value, n_sub_index_values):
            multi_indices[position + (j - sub_index_value), sub_index] = j;

        return position + (n_sub_index_values - sub_index_value);

    # Recursive case.
    else : # if (sub_index < order - 1):
        for j in range(sub_index_value, n_sub_index_values):
            new_position = Recursive_Multi_Indices(
                            multi_indices       = multi_indices,
                            n_sub_index_values  = n_sub_index_values,
                            order               = order,
                            sub_index           = sub_index + 1,
                            sub_index_value     = j,
                            position            = position);

            for k in range(0, (new_position - position)):
                multi_indices[position + k, sub_index] = j;

            position = new_position;

        return position;



def Generate_Library(
        u_NN            : Neural_Network,
        Coords          : torch.Tensor,
        num_derivatives : int,
        PDE_order       : int) -> np.array:

    """ This function populates the library matrix in the SINDY algorithm. How
    this works (and why it works the way that it does) is a tad involved.... so
    buckle up, here comes a (rather verbose) explanation:

    To populate the library, we need to know the set of all polynomial terms
    whose degree is <= order. To determine that set, we essentially need to
    find the set of all "distinct" multi-indices with up to order sub-indices.
    Here, two multi-indices are "equivalent" if the sub-indices in one
    multi-index can be rearranged to match that of the other multi-index (the
    same up to rearrangements of the sub-index ordering).

    Why does this give us what we want? Let's consider when order = 2 and
    num_derivatives = 2. In this case, our library consists of the following:
        order 0: c,
        order 1: u, du_dx, d^2u_dx^2
        order 2: (u)^2, u*du_dx, u*d^2u_dx^2, (du_dx)^2, du_dx*d^2u_dx^2, (d^2u_dx^2)^2
    We want to reframe this as a multi-index problem. In particular, let us
    associate a multi-index to each term above. nth-order terms will consist of
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
    column for each distinct multi-index with at most order sub-indices. We
    then evaluate u, du_dx,... at each point in Coords. If the ith column of
    the library corresponds to the multi-index (1, 3), then we pointwise
    multiply du_dx and du^3_dx^3 and store the product in that column. """

    # We need a sub-index value for each derivative, as well as u itself.
    n_sub_index_values = num_derivatives + 1;

    # Determine how many multi-indicies with k sub-indicies exist for each
    # k in {0, 1,... order}.
    num_multi_indicies = np.empty(PDE_order + 1, dtype = np.int);
    num_multi_indicies[0] = 1;
    for i in range(1, PDE_order + 1):
        num_multi_indicies[i] = Recursive_Counter(
                                    n_sub_index_values  = n_sub_index_values,
                                    order               = i);

    # Use this information to initialize the Library as a tensor of ones.
    # We need everything to be ones because of how we populate this matrix (see
    # below).
    num_rows : int = Coords.shape[0];
    num_cols : int = num_multi_indicies.sum();
    Library : torch.Tensor = np.ones((num_rows, num_cols), dtype = np.float32);

    # Evaluate u, du/dx,... at each point.
    (du_dt, diu_dxi) = Evaluate_u_derivatives(
                            u_NN            = u_NN,
                            num_derivatives = num_derivatives,
                            Coords          = Coords);

    # Now populate the library the multi-index approach described above. Note
    # that the first column corresponds to a constant and should, therefore,
    # be filled with 1's (which it already is).
    position = 1;
    for order in range(1, PDE_order):
        # Create a buffer to hold the multi-indicies.
        multi_indices = np.empty((num_multi_indicies[order], order), dtype = np.int);

        # Find the set of multi indicies!
        Recursive_Multi_Indices(
            multi_indices       = multi_indices,
            n_sub_index_values  = n_sub_index_values,
            order               = order);

        # Cycle through the multi-indicies of this order
        for i in range(num_multi_indicies[order]):

            # Cycle through the sub-indicies of this multi-index.
            for j in range(order):
                Library[:, position] = (Library[:, position]*
                                        diu_dxi[:, multi_indices[i, j]].detach().squeeze().numpy());

            # Increment position
            position += 1;

    # All done, the library is now populated!
    return Library;



def main():
    # Initialize parameters.
    n_sub_index_values  = 4;
    order               = 3;

    # Run Recursive_Counter to determine how big x must be.
    counter = Recursive_Counter(
                n_sub_index_values  = n_sub_index_values,
                order               = order);

    # allocate space for x.
    multi_indices = np.empty((counter, order), dtype = np.int);

    # Populate x using Recursive_Multi_Indices
    Recursive_Multi_Indices(
        multi_indices       = multi_indices,
        n_sub_index_values  = n_sub_index_values,
        order               = order);



    # Print results.
    print(counter);
    print(multi_indices);



if(__name__ == "__main__"):
    main();
