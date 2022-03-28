import numpy;



def Create_Data_Set(        Name            : str,
                            Train_Inputs    : numpy.ndarray,
                            Train_Targets   : numpy.ndarray,
                            Test_Inputs     : numpy.ndarray,
                            Test_Targets    : numpy.ndarray) -> None:
    """ This function generates a DataSet (a file in Data/DataSets) with a
    specified Name, set of inputs, and corresponding set of target values.

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

    ----------------------------------------------------------------------------
    Returns:

    Nothing! """

    # Fist, open the file.
    Path : str = "./DataSets/" + Name + ".npz";
    File = open(Path, mode = "wb");

    # Now, serialize the Inputs and Targets arrays
    numpy.savez(    file = File,
                    Train_Inputs    = Train_Inputs,
                    Train_Targets   = Train_Targets,
                    Test_Inputs     = Test_Inputs,
                    Test_Targets    = Test_Targets);

    # All done!
    File.close();
    return;
