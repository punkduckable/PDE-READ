import numpy;
import matplotlib.pyplot as pyplot;

from Create_Data_Set import Create_Data_Set;



def Static_Beam(File_Name : str):
    """ This function generates a DataSet for the Static Beam experiment from
    CU Bend. This function is literally hard coded to a specific file...

    ----------------------------------------------------------------------------
    Arguments:

    File_Name : The name of the text file containing the data (without
    extension).

    ----------------------------------------------------------------------------
    Returns:

    Nothing! """

    # First, open the file.
    Path : str  = "./" + File_Name + ".txt";
    File        = open(Path, 'r');

    # Each line of this file uses the following format:
    #       a.bcdefge+pq      # Node  k:    y displacement
    # We want to extract the float at the start.
    n               : int           = 11;
    Position        : int           = numpy.empty(shape = (n, 1), dtype = numpy.float32);
    Displacement    : numpy.ndarray = numpy.empty(shape = n, dtype = numpy.float32);

    for i in range(n):
        Position[i, 0]     = 2.0*i;

        Line      : str = File.readline();
        Displacement[i] = 1000*float(Line.split()[0]);


    # Plot what we read (sanity check)
    pyplot.plot(Position, Displacement);
    pyplot.show();

    # Create the data set
    Create_Data_Set(    Name            = "CU_Bend",
                        Train_Inputs    = Position,
                        Train_Targets   = Displacement,
                        Test_Inputs     = Position,
                        Test_Targets    = Displacement);

    # All done.
    File.close();



def main():
    Static_Beam(File_Name = "Y_displacements");


if __name__ == "__main__":
    main();
