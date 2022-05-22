import numpy;



def Create_Data_Set(        Name            : str,
                            Train_Inputs    : numpy.ndarray,
                            Train_Targets   : numpy.ndarray,
                            Test_Inputs     : numpy.ndarray,
                            Test_Targets    : numpy.ndarray,
                            Input_Bounds    : numpy.ndarray) -> None:
    """ This function generates a DataSet (a file in Data/DataSets) with a
    specified Name, set of inputs, target values, and problem domain bounds. We
    assume the inputs are a rectangle in R^n. That is, there is some {a_1, ... ,
    a_n, b_1, ... , b_n} such that each input (test and train) is in the set
    [a_1, b_1] x ... x [a_n, b_n].

    ----------------------------------------------------------------------------
    Arguments:

    Name : This is a string. This function creates a new DataSet. This is the
    name of the file we save the DataSet set to.

    Train_Inputs : This should be a matrix of data. If there are m training
    inputs, each one of which lives in R^n, then this should be a m by n matrix
    whose ith row holds the ith input.

    Train_Targets : This should be a vector of target values. If there are m
    training inputs, then this should be a vector in R^m.

    Test_Inputs, Test_Targets: Same as Train_Inputs and Train_Targets,
    respectively, but for the testing set.

    Input_Bounds : We assume that each input lies in some rectangle, [a_1, b_1]
    x ... x [a_n, b_n]. This argument should be a n by 2 array whose ith row is
    [a_i, b_i].

    ----------------------------------------------------------------------------
    Returns:

    Nothing! """

    # Fist, open the file.
    Path : str = "./DataSets/" + Name + ".npz";
    File = open(Path, mode = "wb");

    # Now, serialize the Inputs and Targets arrays
    numpy.savez(    file            = File,
                    Train_Inputs    = Train_Inputs,
                    Train_Targets   = Train_Targets,
                    Test_Inputs     = Test_Inputs,
                    Test_Targets    = Test_Targets,
                    Input_Bounds    = Input_Bounds);

    # All done!
    File.close();
    return;
