import numpy as np;
import torch;
import math;
from typing import Tuple, List;
from sklearn import linear_model;

from Network import Neural_Network;
from PDE_Residual import Evaluate_Sol_Derivatives;



def Recursive_Counter(
        num_sub_index_values : int,
        num_sub_indices      : int,
        sub_index            : int = 0,
        sub_index_value      : int = 0,
        counter              : int = 0) -> int:
    """ This function determines the number of "distinct" multi-indices of
    specified num_sub_indices whose sub-indices take values in {0, 1...
    num_sub_index_values - 1}. Here, two multi-indices are "equal" if and only
    if there is a way to rearrange the sub-indices in one multi-index to match
    the others (both have the same value in each sub-index). This defines an
    equivalence relation of multi-indices. Thus, we are essentially finding the
    number of classes under this relation.

    For example, if num_sub_index_values = 4 and num_sub_indices = 2, then the
    set of possible multi-indices is { (0, 0), (0, 1), (0, 2), (0, 3), (1, 1),
    (1, 2), (1, 3), (2, 2), (2, 3), (3, 3) }, which contains 10 elements. Thus,
    in this case, this function would return 10.

    Note: we assume that num_sub_indices and num_sub_index_values are POSITIVE
    integers.

    ----------------------------------------------------------------------------
    Arguments:

    num_sub_index_values: The number of distinct values that any one of the
    sub-indices can take on. If num_sub_index_values = k, then each sub-index
    can take on values 0, 1,... k-1.

    num_sub_indices: The number of sub-indices in the multi-index.

    sub_index: keeps track of which sub-index we're working on.

    sub_index_value: specifies which value we put in a particular sub-index.

    counter: stores the total number of multi-indices of specified oder whose
    sub-indices take values in 0, 1... num_sub_index_values - 1. We ultimately
    return this variable. It's passed as an argument for recursion.

    ----------------------------------------------------------------------------
    Returns:

    The total number of "distinct" multi-indices (as defined above) which have
    num_sub_indices sub-indices, each of which takes values in {0, 1,...
    num_sub_index_values - 1}. """

    # Assertions.
    assert (num_sub_indices > 0), \
        ("num_sub_indices must be a POSITIVE integer. Got %d." % num_sub_indices);
    assert (num_sub_index_values > 0), \
        ("num_sub_index_values must be a POSITIVE integer. Got %d." % num_sub_index_values);

    # Base case
    if (sub_index == num_sub_indices - 1):
        return counter + (num_sub_index_values - sub_index_value);

    # Recursive case.
    else : # if (sub_index < num_sub_indices - 1):
        for j in range(sub_index_value, num_sub_index_values):
            counter = Recursive_Counter(
                        num_sub_index_values = num_sub_index_values,
                        num_sub_indices      = num_sub_indices,
                        sub_index            = sub_index + 1,
                        sub_index_value      = j,
                        counter              = counter);

        return counter;



def Recursive_Multi_Indices(
        multi_indices        : np.array,
        num_sub_index_values : int,
        num_sub_indices      : int,
        sub_index            : int = 0,
        sub_index_value      : int = 0,
        position             : int = 0) -> int:
    """ This function finds the set of "distinct" multi-indices with
    num_sub_indices sub-indices such that each sub-index takes values in
    {0, 1,... num_sub_index_values - 1}. Here, two multi-indices are "equal" if
    and only if there is a way to rearrange the sub-indices in one multi-index
    to match the others (both have the same value in each sub-index). This
    defines an equivalence relation. Thus, we return a representative for each
    class.

    We assume that multi_indices is an N by num_sub_indices array, where N is
    "sufficiently large" (meaning that N is at least as large as the value
    returned by Recursive_Counter with the num_sub_index_values and
    num_sub_indices arguments). This function populates the rows of
    multi_indices. The i,j element of multi_indices contains the value of the
    jth sub-index of the ith "distinct" (as defined above) multi-index.

    For example, if num_sub_index_values = 4 and num_sub_indices = 2, then the
    set of "distinct" multi-indices is { (0, 0), (0, 1), (0, 2), (0, 3), (1, 1),
    (1, 2), (1, 3), (2, 2), (2, 3), (3, 3) }. This function will populate the
    first 10 rows of multi_indices as follows:
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

    Note: we assume that num_sub_indices and num_sub_index_values are POSITIVE
    integers.

    ----------------------------------------------------------------------------
    Arguments:

    multi_indices: An N by num_sub_indices tensor, where N is "sufficiently
    large" (see above). This array will hold all distinct multi-indices with
    num_sub_indices sub-indices, each of which takes values in {0, 1,...
    num_sub_index_values - 1}.

    num_sub_index_values: The number of distinct values that any sub-index can
    take on. If num_sub_index_values = k, then each sub_index can take values
    0, 1,... num_sub_index_values - 1.

    num_sub_indices: The number of sub-indices in the multi-index.

    sub_index: keeps track of which sub-index we're working on.

    sub_index_value: specifies which value we put in a particular sub-index.

    ----------------------------------------------------------------------------
    Returns:

    An integer that the function uses for recursion. You probably want to
    discard it. """

    # Assertions.
    assert (num_sub_indices > 0), \
        ("num_sub_indices must be a POSITIVE integer. Got %d." % num_sub_indices);
    assert (num_sub_index_values > 0), \
        ("num_sub_index_values must be a POSITIVE integer. Got %d." % num_sub_index_values);

    # Base case
    if (sub_index == num_sub_indices - 1):
        for j in range(sub_index_value, num_sub_index_values):
            multi_indices[position + (j - sub_index_value), sub_index] = j;

        return position + (num_sub_index_values - sub_index_value);

    # Recursive case.
    else : # if (sub_index < num_sub_indices - 1):
        for j in range(sub_index_value, num_sub_index_values):
            new_position = Recursive_Multi_Indices(
                            multi_indices        = multi_indices,
                            num_sub_index_values = num_sub_index_values,
                            num_sub_indices      = num_sub_indices,
                            sub_index            = sub_index + 1,
                            sub_index_value      = j,
                            position             = position);

            for k in range(0, (new_position - position)):
                multi_indices[position + k, sub_index] = j;

            position = new_position;

        return position;



def Generate_Library(
        Sol_NN          : Neural_Network,
        PDE_NN          : Neural_Network,
        Coords          : torch.Tensor,
        num_derivatives : int,
        Poly_Degree     : int,
        Torch_dtype     : torch.dtype = torch.float32,
        Device          : torch.device = torch.device('cpu')) -> Tuple[np.array,
                                                                       np.array,
                                                                       np.array,
                                                                       List]:
    """ This function populates the "library" matrix. Let u denote the function
    that Sol_NN approximates. We assume that u satisifies the following:
            (d/dt)u = c_{0,0} + c_{1,0}u + c_{1,1}(d/dx)u +... c_{n,1}(d^n/dx^n)u
                    + c_{1,0}u*u + c_{1,1}u*(d/dx)u +....                     {a}
    Where each c_{i,j} is a constant. Notice that the right side of the above
    equation is a polynomial of u, (d/dx)u,... (d^n/dx^n)u. Each term consisits
    of two parts, the constant part (c_{i,j}) and the "u part" (the product of u
    and its derivatives). Poly_Degree specifies the maximum degree of the
    polynomial terms. num_derivatives determines the highest order derivative of
    u that is present in the polynomial terms.

    To find the constants, we evaluate u, (d/dx)u,... (d^n/dx^n)u at the Coords.
    We use these values to construct the "u part" of each polynomial term.
    This gives us a set of linear equations (in the constants), one for each
    row of Coords. We then try to find the constants which satisfy
                [(d/dt)u] = L(u)*c
    in a least-squares sense. where the ith element of [(d/dt)u] holds the value
    of (d/dt)u at the ith Coord. The i,j entry of L(u) holds the value of the
    "u part" of the jth polynomial term in {a} evaluated at the ith Coord. c is
    a vector holding the coefficients.

    This function constructs L(u) and du/dt. We use the Recursive_Multi_Indices
    function to find the set of polynomial terms. Why does this work? Let's
    consider when Poly_Degree = 2 and num_derivatives = 2. Let u = Sol_NN. In
    this case, the "u parts" of the polynomial terms are the following:
        degree 0: 1,
        degree 1: u, (d/dx)u, (d^2/dx^2)u
        degree 2: (u)^2, u*(d/dx)u, u*(d^2/dx^2)u, ((d/dx)u)^2, (d/dx)u*(d^2/dx^2)u, ((d^2/dx^2)u)^2
    Notice that we can associate each term of degree n with a multi-index with
    n sub-indices. The ith term in the "u part" gets associated with the ith
    sub-index. In particular, we can associate u with 0, (d/dx)u with 1, and
    (d^2/dx^2)u with 2. Then, for example, we associate the term u*du_dx with
    the multi-index (0, 1). Notice that, since multiplication commutes, only one
    of u*(d/dx)u, (d/dx)u*u appears in {a}. Thus, the multi-indices (1, 0) and
    (0, 1) are "the same". However, this is precisely my definition of
    multi-index equivalence above.

    Once we know the set of all "distinct" multi-indices, we can construct L(u).
    In particular, we first allocate an array that has a column for each distinct
    multi-index with up to Poly_Degree sub-indices, each of which takes values
    in {0, 1, 2,... num_derivatives}. We then evaluate u, du_dx,... at each point
    in Coords. If the jth column L(u) corresponds to the multi-index (1, 3),
    then it will hold the element-wise product of (d/dx)u and (d^3/dx^3)u.

    The columns of L(u) are ordered according to what's returned by
    Recursive_Multi_Indices.

    Note: This function only works if u depends on one spatial variable.

    ----------------------------------------------------------------------------
    Arguments:

    Sol_NN: The network that approximates the PDE solution.

    PDE_NN: The network that approximates the PDE.

    Coords: The coordinates of the extraction points (The library will contain
    a row for each element of Coords).

    num_derivatives: The highest order derivative that we think is in the
    governing PDE.

    Poly_Degree: the maximum degree of the polynomial terms in {a}. For
    example, if we expect to extract a linear PDE, then Poly_Degree should be 1.
    Setting Poly_Degree > 1 allows the algorithm to search for non-linear PDEs.

    Torch_dtype: The data type that all tensors in Sol_NN and PDE_NN use.

    Device: The device that Sol_NN and PDE_NN are loaded on.

    ----------------------------------------------------------------------------
    Returns:

    A 4 element tuple. For the first, we evaluate u, u_xx,... at each point in
    Coords. We then evaluate PDE_NN at each point. This is the first element of
    the returned tuple. The second holds the library (with a row for each
    coordinate and a column for each term in the library). The third holds a
    Poly_Degree element array whose ith element holds the number of
    multi_indices with k sub-indices, each of which take values in {0, 1,...
    num_derivatives}. The 4th is a list of arrays, the ith one of which holds
    the set of multi-indices of order i whose sub-indices take values in
    {0, 1,... num_derivatives}. """

    # We need a sub-index value for each derivative, as well as u itself.
    num_sub_index_values = num_derivatives + 1;

    # Determine how many multi-indices exist with num_sub_indices sub-indices
    # such that each sub-index takes values in {0, 1,... Poly_Degree}.
    num_multi_indices    = np.empty(Poly_Degree + 1, dtype = np.int64);
    num_multi_indices[0] = 1;
    for i in range(1, Poly_Degree + 1):
        num_multi_indices[i] = Recursive_Counter(
                                    num_sub_index_values = num_sub_index_values,
                                    num_sub_indices      = i);

    # Set up a list to hold the multi-index arrays for different number of
    # sub-indices.
    multi_indices_list = [];
    multi_indices_list.append(np.array(1, dtype = np.int64));

    # Initialize the Library. We initialize with ones because of how we populate
    # the library (see below).
    num_rows : int = Coords.shape[0];
    num_cols : int = num_multi_indices.sum();
    Library : np.array = np.ones((num_rows, num_cols), dtype = np.float32);

    # Evaluate u, du/dx,... at each point. We use batches to reduce memory load.
    du_dt   = torch.empty(  (num_rows),
                            dtype  = Torch_dtype,
                            device = Device);
    diu_dxi = torch.empty(  (num_rows, num_sub_index_values),
                            dtype  = Torch_dtype,
                            device = Device);

    # Main loop
    Batch_Size : int = 1000;
    for i in range(0, num_rows - Batch_Size, Batch_Size):
        (du_dt_Batch, diu_dxi_Batch) = Evaluate_Sol_Derivatives(
                                Sol_NN          = Sol_NN,
                                num_derivatives = num_derivatives,
                                Coords          = Coords[i:(i + Batch_Size)],
                                Data_Type       = Torch_dtype,
                                Device          = Device);

        du_dt[i:(i + Batch_Size)]   = du_dt_Batch.detach();
        diu_dxi[i:(i + Batch_Size)] = diu_dxi_Batch.detach();

    # Clean up loop.
    (du_dt_Batch, diu_dxi_Batch) = Evaluate_Sol_Derivatives(
                                        Sol_NN          = Sol_NN,
                                        num_derivatives = num_derivatives,
                                        Coords          = Coords[(i + Batch_Size):],
                                        Data_Type       = Torch_dtype,
                                        Device          = Device);

    du_dt[(i + Batch_Size):]   = du_dt_Batch.detach();
    diu_dxi[(i + Batch_Size):] = diu_dxi_Batch.detach();

    # Evaluate PDE_NN at the output given by diu_dxi.
    PDE_NN_At_Coords = PDE_NN(diu_dxi).detach().cpu().squeeze().numpy().astype(dtype = np.float32);

    # Now populate the library using the multi-index approach described above.
    # Note that the first column corresponds to a constant and thus should
    # contain 1's (which it already does).
    position = 1;
    for k in range(1, Poly_Degree + 1):
        # Create a buffer to hold the multi-indices of with k sub-indices.
        multi_indices = np.empty((num_multi_indices[k], k), dtype = np.int64);

        # Find the set of multi-indices with k sub-indices.
        Recursive_Multi_Indices(
            multi_indices        = multi_indices,
            num_sub_index_values = num_sub_index_values,
            num_sub_indices      = k);

        # Cycle through the multi-indices with k sub-indices.
        for i in range(num_multi_indices[k]):

            # Cycle through the sub-indices of this multi-index.
            for j in range(k):
                Library[:, position] = (Library[:, position]*
                                        diu_dxi[:, multi_indices[i, j]].detach().cpu().squeeze().numpy().astype(dtype = np.float32));

            # Increment position
            position += 1;

        # Append this set of multi-indices to the list.
        multi_indices_list.append(multi_indices);

    # We've populated the library! Package everything together and return.
    return (PDE_NN_At_Coords,
            Library,
            num_multi_indices,
            multi_indices_list);



def Recursive_Feature_Elimination(
        A             : np.array,
        b             : np.array) -> Tuple[np.array, np.array]:
    """ This function repeatedly solves Ax = b in the the least-squares sense.
    However, after each solve, it eliminates (sets to zero) the least important
    feature (component of x), giving a sequence of progressively sparser least
    squares solutions. We call these solutions "candidates". At each step, we
    record the candidate solution, x, and the value ||Ax - b||_2.
        So how do we pick which feature to eliminate? We try to pick the one
    that, if removed, would cause the residual to increase the least. In an
    ideal world, we could identify this feature using the following algorithm:
        for each feature x_k, find the least-squares solution to Ax = b with
        x_k = 0, evaluate the residual. Pick the feature which corresponds to
        the smallest residual.
    Doing this, however, requires O(n^2) least square solutions, which can get
    very expensive very quickly. Instead, we remove the feature whose magnitude
    is the smallest (which we call the "least important feature"). This
    criterion is mathematically justified if each of A's columns has a unit L2
    norm. Why? Well, the residual is a smooth quadratic function of each
    feature. Since the least-squares solution is the global minima of the
    Residual, we have
        R(x - x_k e_k ) - R(x) = (1/2)(d^2R(x)/dx_k^2)(x_k^2)
    This tells us how much the residual would change if eliminate the kth
    feature but held the others fixed. In reality, the value we want is
    ||Ax' - b||_2, where x' is the least-squares solution to Ax' = b s.t.
    x'_k = 0. In general, x' != x - x_k e_k. However, we expect that
    x' ~= x - x_k e_k, which means that R(x - x_k e_k) only approximates R(x').
    We do, however, expect this approximation to be decent. A little algebra
    shows that (d^2R(x)/dx_k^2) is the square of the L2 norm of A's kth column.
    Thus, if each column of A has a unit L2 norm, R(x - x_k e_k) - R(x) =
    (1/2)(x_k^2). Therefore, in this case, the least important feature is the
    one with the smallest magnitude.

    This function first scales the columns of A to have a unit L2 norm. It
    solves Ax = b in the least-squares sense, selects the feature with the
    smallest magnitude, and then repeats with that feature removed.

    ----------------------------------------------------------------------------
    Arguments:

    A: The 'A' in Ax = b

    b: The 'b' in Ax = b.

    ----------------------------------------------------------------------------
    Returns :

    A two-element tuple. The first element holds a matrix, X. If A is an n by m
    matrix, then X is an m by m matrix whose jth column holds the jth candidate
    solution vector. The second element is a vector, Residual, whose jth element
    holds the residual of the jth candidate solution. """

    # First, determine the number of columns of A, which determines the number
    # of iterations.
    Num_Rows : int = A.shape[0];
    Num_Cols : int = A.shape[1];

    # Scale each column of L to have a unit L2 norm. This guarantees that the
    # component of the solution with the smallest magnitude is the least
    # salient in the sense that removing it will lead to the smallest increase
    # in residual.
    A_Column_L2_Norms      = np.empty(Num_Cols, dtype = np.float32);
    A_Scaled              = np.empty((Num_Rows, Num_Cols), dtype = np.float32);
    for j in range(Num_Cols):
        # Sum up the squares of the entries of the jth column of A.
        Sum = 0;
        for i in range(Num_Rows):
            Sum += A[i, j]*A[i, j];

        # Determine L2 column norm, scale that column of A.
        A_Column_L2_Norms[j] = math.sqrt(Sum);
        A_Scaled[:, j] = A[:, j]/A_Column_L2_Norms[j];

    # Initialize the residual and feature lists. There's an extra element in the
    # Residual Array for the L2 norm of b (the solution with no features).
    X                   = np.zeros((Num_Cols, Num_Cols), dtype = np.float32);
    Remaining_Features  = np.ones(Num_Cols, dtype = np.bool);
    Residual            = np.empty(Num_Cols + 1, dtype = np.float32);

    # Recursively eliminate features based on the magnitude of the weights.
    for j in range(Num_Cols):
        # Compute the least-squares solution, store it in the ith row of X.
        X[Remaining_Features, j] = np.linalg.lstsq(A_Scaled[:, Remaining_Features], b, rcond = None)[0];

        # Now, determine the smallest remaining component of the solution.
        for p in range(Num_Cols):
            if(Remaining_Features[p] == True):
                break;

        Index_Smallest = p;
        for q in range(Num_Cols):
            if(Remaining_Features[q] == True and (abs(X[q, j]) < abs(X[Index_Smallest, j]))):
                Index_Smallest = q;

        Remaining_Features[Index_Smallest] = False;

        # Now, evaluate the Residual (L2 norm of b - Ax)
        Diff = b - A_Scaled @ X[:, j];
        Sum = 0;
        for i in range(Num_Rows):
            Sum += Diff[i]*Diff[i];
        Residual[j] = math.sqrt(Sum);

    # Finally, evaluate the residual with all features removed.
    Sum = 0;
    for i in range(Num_Rows):
        Sum += b[i]*b[i];
    Residual[Num_Cols] = math.sqrt(Sum);

    # Now, we need to scale the components of X by the corresponding norms of
    # the columns of A. Why? We solved A_Scaled*x ~= b in the least-squares
    # sense. The jth column of A_Scaled is the jth column of A divided by the L2
    # norm of that column. Thus, if x' is the vector such that
    # x'_j = x_j/||A_j||_2 (where ||A_j||_2 is the L2 norm of the jth column of
    # A), then x' satisfies Ax' ~= b in the least-squares sense.
    for j in range(Num_Cols):
        for i in range(Num_Cols):
            X[i, j] /= A_Column_L2_Norms[i];

    return X, Residual;



def Rank_Candidate_Solutions(
        X        : np.array,
        Residual : np.array) -> Tuple[np.array, np.array, np.array]:
    """ This function ranks the candidates from "Recursive_Feature_Elimination"
    by how much the residual increases (in a relative sense) if we remove the
    least important term from that candidate.

    ----------------------------------------------------------------------------
    Arguments :

    X, Residual: the outputs of Recursive_Feature_Elimination.

    ----------------------------------------------------------------------------
    Returns :

    A three-element tuple. The first element holds X_Ranked, a copy of X whose
    columns are ranked according to how likely they are the true solution.
    If X is an m by m matrix, then X_ranked is as well. The 0 column of X_Ranked
    is the least likely solution, while the last column of X_ranked is the most
    likely one. The second element is Residual_Ranked, a reordered version
    of Residual such that the jth element of Residual_Ranked gives the residual
    of the solution specified in the jth column of X_ranked. The third element
    is a vector, Residual_Change, whose jth element specifies how much the
    residual increases (as a percentage) when we remove the least important
    feature from the solution specified in the jth column of X_ranked. """

    # First, determine the relative change in residual after each step.
    Num_Cols = Residual.size - 1;
    Residual_Change = np.empty(Num_Cols, dtype = np.float32);
    for i in range(Num_Cols):
        Residual_Change[i] = (Residual[i + 1] - Residual[i])/Residual[i];

    # Initialize an array to keep track of the largest components of
    # Residual_Change.
    Index_Rankings = np.empty(Num_Cols, dtype = np.int64);
    for i in range(Num_Cols):
        Index_Rankings[i] = i;

    # Now, figure out how to sort the elements of Residual_Change.
    for i in range(Num_Cols):
        # Determine the index of the smallest remaining component of
        # Residual_Change.
        Index_Smallest = i;
        for j in range(i, Num_Cols):
            if(Residual_Change[j] < Residual_Change[Index_Smallest]):
                Index_Smallest = j;

        # Now swap the ith and Index_Smallest components of Residual_Change.
        # Note the change in Index_Rankings.
        temp = Residual_Change[i];
        Residual_Change[i] = Residual_Change[Index_Smallest];
        Residual_Change[Index_Smallest] = temp;

        temp = Index_Rankings[i];
        Index_Rankings[i] = Index_Rankings[Index_Smallest];
        Index_Rankings[Index_Smallest] = temp;

    # Now, reorder the columns of X and the components of Residual according to
    # Index_Rankings.
    X_Ranked        = np.empty_like(X, dtype = np.float32);
    Residual_Ranked = np.empty(Num_Cols, dtype = np.float32);

    for j in range(Num_Cols):
        X_Ranked[:, j]     = X[:, Index_Rankings[j]];
        Residual_Ranked[j] = Residual[Index_Rankings[j]];

    # All done!
    return (X_Ranked, Residual_Ranked, Residual_Change);



def Print_Extracted_PDE(
        Extracted_PDE      : np.array,
        num_multi_indices  : np.array,
        multi_indices_list : Tuple) -> None:
    """ This function takes in the output of Thresholded_Least_Squares and
    prints it as a human-readable PDE.

    Note: this function only works if u (the solution) is a function of 1
    spatial variable.

    ----------------------------------------------------------------------------
    Arguments:

    Extracted_PDE: This should be the thresholded least-squares solution
    (returned by Thresholded_Least_Squares).

    num_multi_indices, multi_indices_list: the corresponding variables returned
    by Generate_Library.

    ----------------------------------------------------------------------------
    Returns:

    Nothing! """

    # Start the printout.
    print("du/dt =", end = '');

    if (Extracted_PDE[0] != 0):
        print("%f " % Extracted_PDE[0], end = '');

    position : int    = 1;
    for k in range(1, num_multi_indices.shape[0]):
        # Cycle through the multi-indices with k sub-indices.
        for i in range(num_multi_indices[k]):

            # If this term of the extracted PDE is non-zero, print out the
            # corresponding term.
            if(Extracted_PDE[position] != 0):
                print(" + %f" % Extracted_PDE[position], end = '');
                multi_index = multi_indices_list[k][i];

                # cycle through the sub-indices of this multi-index.
                for j in range(k):
                    if(multi_index[j] == 0):
                        print("(u)", end = '');
                    else:
                        print("(d^%d u/dx^%d)" % (multi_index[j], multi_index[j]), end = '');
            position += 1;

    # Finish printing (this prints a newline character).
    print();
