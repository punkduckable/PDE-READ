class Read_Error(Exception):
    # Raised if we can't find a Phrase in a File.
    pass;



class Setup_Data_Container:
    # A container for data read in from the setup file.
    pass;



def Index_After_Phrase(Line_In : str, Phrase_In : str, Case_Sensitive : bool = False) -> int:
    """ This function searches for Phrase_In within Line_In. If found, it
    returns the index of the first character in Line_In after the first instance
    of Phrase_In within Line_In.

    ----------------------------------------------------------------------------
    Arguments:

    Line_In: A string that contains the line of text. The program searches for
    Phrase_In within Line_In.

    Phrase_In: A string containing the phrase we're searching for.

    Case_Sensitive: Controls if the search is case sensitive or not. (see
    Read_Line_After's doc string for more detials).

    ----------------------------------------------------------------------------
    Returns:

    If we can find Phrase_In within Line_In, then this returns the index of the
    first character in Line_In after the first instance of Phrase_In within
    Line_In. If we can't find Phrase_In within Line_In, then this function
    returns -1. """

    # First, get the number of characters in Line, Phrase. We will use these as
    # loop bounds.
    Num_Chars_Line : int = len(Line_In);
    Num_Chars_Phrase : int = len(Phrase_In);

    # If we should ignore case, then map Phrase, Line to lower case versions of
    # themselves. Note that we perform this operation on a copy of the input
    # Line/Phrase so that we don't modify the passes variables.
    Line = Line_In;
    Phrase = Phrase_In;
    if(Case_Sensitive == False):
        Line = Line.lower();
        Phrase = Phrase.lower();

    # If Phrase is in Line, then the first character of Phrase must occur before
    # the Num_Chars_Line - Num_Chars_Phrase character of Line (think about it).
    # Thus, we only need to loop up to the Num_Chars_Line - Num_Chars_Phrase
    # character of Line.
    for i in range(Num_Chars_Line - Num_Chars_Phrase + 1):
        # Check if ith character of Line_Copy matches the 0th character of Phrase. If
        # so, check for a match. This happens if for each j in 0,...
        # Num_Chars_Phrase - 1, Line[i + j] == Phrase[j] (think about it).
        if(Line[i] == Phrase[0]):
            Match : Bool = True;

            for j in range(1, Num_Chars_Phrase):
                # If Line[i + j] != Phrase[j], then we do not have a match, we
                # should move onto the next character of Line.
                if(Line[i + j] != Phrase[j]):
                    Match = False;
                    break;

            # If Match is still True, then i+ Num_Chars_Phrase is the index of
            # the first character in Line beyond Phrase.
            if(Match == True):
                return i + Num_Chars_Phrase;

    # If we get here, then we made it through Line without finding Phrase. Thus,
    # Phrase is not in Line. We return -1 to indiciate that.
    return -1;



def Read_Line_After(File, Phrase : str, Case_Sensitive = False) -> str:
    """ This function tries to find Phrase in a line of File. In particular,
    it searches through the lines of File until it finds an instance of Phrase.
    When it finds Phrase in one of File's lines, it returns everything after
    the Phrase in that line. If it can't find the Phase in one of File's lines,
    it raises an exception.

    ----------------------------------------------------------------------------
    Arguments:

    File: The file in which we want to search for Phrase.

    Phrase: The Phrase we want to find.

    Case_Sensitive: Controls if the search is case sensitive or not. If
    True, then we search for an exact match (including case) of Phrase in one of
    File's lines. If not, then we try to find a line of File which contains the
    same letters (in the same order) as Phrase.

    ----------------------------------------------------------------------------
    Returns:

    Everything after Phrase in the first line of File that contains Phrase.
    Thus, if the Phrase is "cat is", and one of File's lines is "the cat is fat"
    , then this will return " fat". """

    # Cycle through the lines of the file until we find one that matches
    # the phrase.
    while(True):
        # Get the next line
        Line = File.readline();

        # Python doesn't use end of file characters. However, readline will
        # only retun an empty string if we're at the end of file. Thus, we can
        # use this as our "end of file" check
        if(Line == ""):
            raise Read_Error("Could not find \"" + Phrase + "\" in File.");

        # If the line is a comment (starts with '#'), then ignore it.
        if (Line[0] == "#"):
            continue;

        # Search for the phrase in Line. If it's in the line, this will
        # return the index of the first character after phrase in Line.
        # If Phrase is not in Line, this will return -1, indiciating that
        # we should check the next line. Otherwise, we should return everything
        # in Line after the returned Index.
        Index : int = Index_After_Phrase(Line, Phrase, Case_Sensitive);
        if(Index == -1):
            continue;
        else:
            return Line[Index:];



def Setup_File_Reader() -> Setup_Data_Container:
    """ This function reads the settings in Setup.txt.

    ----------------------------------------------------------------------------
    Arguments:

    None!

    ----------------------------------------------------------------------------
    Returns:

    A Setup_Data_Container object which contains all of the setings read in
    from Setup.txt. The main function uses these to set up the program. """

    # Open file, initialze a Setup_Data object.
    File = open("../Setup.txt", "r");
    Setup_Data = Setup_Data_Container();



    ############################################################################
    # General Settings

    # Load u network state from File?
    Buffer = Read_Line_After(File, "Load u Network State [bool] :").strip();
    if  (Buffer[0] == 'T' or Buffer[0] == 't'):
        Setup_Data.Load_u_Network_State = True;
    elif(Buffer[0] == 'F' or Buffer[0] == 'f'):
        Setup_Data.Load_u_Network_State = False;
    else:
        raise Read_Error("\"Load u Network State\" should be \"True\" or \"False\". Got " + Buffer);

    # Load N network state from File?
    Buffer = Read_Line_After(File, "Load N Network State [bool] :").strip();
    if  (Buffer[0] == 'T' or Buffer[0] == 't'):
        Setup_Data.Load_N_Network_State = True;
    elif(Buffer[0] == 'F' or Buffer[0] == 'f'):
        Setup_Data.Load_N_Network_State = False;
    else:
        raise Read_Error("\"Load N Network State\" should be \"True\" or \"False\". Got " + Buffer);

    # Load optimizer state from file?
    Buffer = Read_Line_After(File, "Load Optimizer State [bool] :").strip();
    if  (Buffer[0] == 'T' or Buffer[0] == 't'):
        Setup_Data.Load_Optimize_State = True;
    elif(Buffer[0] == 'F' or Buffer[0] == 'f'):
        Setup_Data.Load_Optimize_State = False;
    else:
        raise Read_Error("\"Load Optimizer State\" should be \"True\" or \"False\". Got " + Buffer);

    # If we are loading anything, get load file name.
    if(     Setup_Data.Load_u_Network_State == True or
            Setup_Data.Load_N_Network_State == True or
            Setup_Data.Load_Optimize_State == True):

        Setup_Data.Load_File_Name = Read_Line_After(File, "Load File Name [str] :").strip();


    # Save to file?
    Buffer = Read_Line_After(File, "Save State [bool] :").strip();
    if(Buffer[0] == 'T' or Buffer[0] == 't'):
        Setup_Data.Save_To_File = True;
    elif(Buffer[0] == 'F' or Buffer[0] == 'f'):
        Setup_Data.Save_To_File = False;
    else:
        raise Read_Error("\"Save State\" should be \"True\" or \"False\". Got " + Buffer);

    # If so, get save file name.
    if(Setup_Data.Save_To_File == True):
        Setup_Data.Save_File_Name = Read_Line_After(File, "Save File Name [str] :").strip();


    # Should we plot the final results?
    Buffer = Read_Line_After(File, "Plot Final Result [bool] :").strip();
    if  (Buffer[0] == 'T' or Buffer[0] == 't'):
        Setup_Data.Plot_Final_Results = True;
    elif(Buffer[0] == 'F' or Buffer[0] == 'f'):
        Setup_Data.Plot_Final_Results = False;
    else:
        raise Read_Error("\"Plot Final Result\" should be \"True\" or \"False\". Got " + Buffer);



    ############################################################################
    # Mode

    # PINNS, PDE Discovery, or PDE Extraction mode?
    Buffer = Read_Line_After(File, "PINNs, Discovery, or Extraction mode [str] :").strip();
    if  (Buffer[0] == 'P' or Buffer[0] == 'p'):
        Setup_Data.Mode = "PINNs";
    elif(Buffer[0] == 'D' or Buffer[0] == 'd'):
        Setup_Data.Mode = "Discovery";
    elif(Buffer[0] == 'E' or Buffer[0] == 'e'):
        Setup_Data.Mode = "Extraction";
    else:
        raise Read_Error("\"PINNs, Discovery, or Extraction mode\" should be \"PINNs\", \"Discovery\", or \"Extraction\". Got " + Buffer);

    # PINNs mode specific settings
    if(Setup_Data.Mode == "PINNs"):
        Setup_Data.Periodic_BCs_Highest_Order   = int(Read_Line_After(File, "Periodic BCs highest order [int] :").strip());
        Setup_Data.Num_Train_Colloc_Points = int(Read_Line_After(File, "Number of Training Collocation Points [int] :").strip());
        Setup_Data.Num_Test_Colloc_Points  = int(Read_Line_After(File, "Number of Testing Collocation Points [int] :").strip());

    # Discovery mode specific settings
    if(Setup_Data.Mode == "Discovery"):
        Setup_Data.Num_Train_Data_Points   = int(Read_Line_After(File, "Number of Data Training Points [int] :").strip());
        Setup_Data.Num_Test_Data_Points    = int(Read_Line_After(File, "Number of Data Testing Points [int] :").strip());
        Setup_Data.Num_Train_Colloc_Points = int(Read_Line_After(File, "Number of Training Collocation Points [int] :").strip());
        Setup_Data.Num_Test_Colloc_Points  = int(Read_Line_After(File, "Number of Testing Collocation Points [int] :").strip());

    # Extraction mode specific settings.
    if (Setup_Data.Mode == "Extraction"):
        Setup_Data.Extracted_PDE_Order   = int(Read_Line_After(File, "Extracted PDE order [int] :").strip());
        Setup_Data.Num_Extraction_Points = int(Read_Line_After(File, "Number of Extraction Points [int] :").strip());



    ############################################################################
    # Network Architecture

    # Load u's network Architecture
    Setup_Data.u_Num_Hidden_Layers = int(Read_Line_After(File, "u Network - Number of Hidden Layers [int] :").strip());
    Setup_Data.u_Neurons_Per_Layer = int(Read_Line_After(File, "u Network - Neurons per Hidden Layer [int] :").strip());

    # Load N's network Architecture
    Setup_Data.N_Num_u_derivatives = int(Read_Line_After(File, "N Network - Number u derivatives [int] :").strip());
    Setup_Data.N_Num_Hidden_Layers = int(Read_Line_After(File, "N Network - Number of Hidden Layers [int] :").strip());
    Setup_Data.N_Neurons_Per_Layer = int(Read_Line_After(File, "N Network - Neurons per Hidden Layer [int] :").strip());



    ############################################################################
    # Network hyper-parameters

    Setup_Data.Epochs = int(Read_Line_After(File, "Number of Epochs [int] :").strip());
    Setup_Data.Learning_Rate = float(Read_Line_After(File, "Learning Rate [float] :").strip());



    ############################################################################
    # Data

    Setup_Data.Data_File_Name = Read_Line_After(File, "Data File Name [str] :").strip();

    # All done! Return the setup data.
    return Setup_Data;
