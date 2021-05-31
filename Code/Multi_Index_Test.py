import numpy as np;

def Recursive_Counter(
        sub_index_max   : int,
        order           : int,
        sub_index       : int = 0,
        sub_index_value : int = 0,
        counter     : int = 0) -> int:
    """ This function determines the number of "distinct" multi-indicies of
    specified order whose sub-indicies take values in 0, 1... index_max - 1.
    Here, two multi-indicies are "the same" if and only if the indicies in one
    of them can be rearranged into the other. This defines an equivalence
    relation of multi-indicies. Thus, we are essentially finding the number of
    equivalence classes.

    For example, if sub_index_max = 4 and order = 2, then the set of possible
    multi-indicies is { (0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3),
    (2, 2), (2, 3), (3, 3) }, which contains 10 elements. Thus, in this case,
    this function would return 10.

    ----------------------------------------------------------------------------
    Arguments:

    sub_index_max : The number of possible variables that can go into each slot.
    For example, if you want to construct polynomial terms of order 3 from the
    variables x, y, and z, then index_max = 3.

    order : the order of the multi-indicies (the number of indicies in each
    multi-index).

    sub_index : keeps track of which sub-index of the multi-index we're working
    on.

    sub_index_value : this specifies which value we want to put in a particular
    sub-index of a muti-index.

    counter : the variable that actually stores the total number of
    multi-indicies of spcieifed oder whose sub-indicies take values in 0, 1...
    sub_index_max - 1. This is what's eventually returned.

    ----------------------------------------------------------------------------
    Returns:

    The total number of "distinct" multi-indicies (as defined above) of
    specified order such that each sub-index takes values in 0, 1...
    sub_index_max - 1. """


    if (sub_index == order - 1):
        return counter + (sub_index_max - sub_index_value);

    else : # if (sub_index < order - 1):
        for j in range(sub_index_value, sub_index_max):
            counter = Recursive_Counter(
                        sub_index_max   = sub_index_max,
                        order           = order,
                        sub_index       = sub_index + 1,
                        sub_index_value = j,
                        counter         = counter);

        return counter;



def Recursive_Multi_Indicies(
        multi_indicies  : np.array,
        sub_index_max   : int,
        order           : int,
        sub_index       : int = 0,
        sub_index_value : int = 0,
        position        : int = 0) -> int:
    """ This function essentially finds the set of "distinct" multi-indicies of
    specified order such that each sub-index in the multi-index takes values in
    0,1... sub_index_max-1. Here, two multi-indicies are "the same" if and only
    if the indicies in one of them can be rearranged into the other. This
    defines an equivalence relation of multi-indicies. Thus, we are essentially
    finding a representative for each equivalence class under this relation.

    We assume that multi_indicies is a N by order array, where N is
    "sufficiently large" (meaning that N is at least as large as the value
    returned by Recursive_Counter with the sub_index_max and order arguments).
    This function each row of multi_indicies with a multi-index. The i, j
    element of multi_indicies contains the value of the jth sub-index of the ith
    "distinct" (as defined above) multi-index.

    For example, if sub_index_max = 4 and order = 2, then the set of "distinct"
    multi-indicies is { (0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3),
    (2, 2), (2, 3), (3, 3) }. This function will populate the first 10 rows of
    multi_indicies with the following:
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

    ----------------------------------------------------------------------------
    Arguments:

    multi_indicies : An "sufficiently large" array which will hold all distinct
    multi-indicies of specified order whose indicies take values in 0, 1,...
    sub_index_max - 1.

    sub_index_max : The maximum allowed value of any one of the indiciies in a
    multi-index.

    order : the order of the multi-indicies (the number of indicies in each
    multi-index).

    sub_index : keeps track of which sub-index of the multi-index we're working
    on.

    sub_index_value : this specifies which value we want to put in a particular
    sub-index of a muti-index.

    ----------------------------------------------------------------------------
    Returns:

    an interger used for recursive purposes. You probably want to discard it. """


    if(sub_index == order - 1):
        for j in range(sub_index_value, sub_index_max):
            multi_indicies[position + (j - sub_index_value), sub_index] = j;

        return position + (sub_index_max - sub_index_value);

    else : # if (sub_index < order - 1):
        for j in range(sub_index_value, sub_index_max):
            new_position = Recursive_Multi_Indicies(
                            multi_indicies  = multi_indicies,
                            sub_index_max   = sub_index_max,
                            order           = order,
                            sub_index       = sub_index + 1,
                            sub_index_value = j,
                            position        = position);

            for k in range(0, (new_position - position)):
                multi_indicies[position + k, sub_index] = j;

            position = new_position;

        return position;



def main():
    # Initialize parameters.
    sub_index_max = 4;
    order         = 4;

    # Run Recursive_Counter to determine how big x must be.
    counter = Recursive_Counter(
                sub_index_max   = sub_index_max,
                order           = order);

    # allocate space for x.
    multi_indicies = np.empty((counter, order), dtype = np.int);

    # Populate x using Recursive_Multi_Indicies
    Recursive_Multi_Indicies(
        multi_indicies  = multi_indicies,
        sub_index_max   = sub_index_max,
        order           = order);

    # Print results.
    print(counter);
    print(multi_indicies);



if(__name__ == "__main__"):
    main();
