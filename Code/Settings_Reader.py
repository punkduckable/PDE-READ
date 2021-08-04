import torch;
import numpy;



class Read_Error(Exception):
    # Raised if we can't find a Phrase in a File.
    pass;



class Settings_Container:
    # A container for data read in from the settings file.
    pass;



def Index_After_Phrase(Line_In : str, Phrase_In : str, Case_Sensitive : bool = False) -> int:
    """ This function searches for the substring Phrase_In within Line_In.

    ----------------------------------------------------------------------------
    Arguments:

    Line_In: A string. The program searches for the substring Phrase_In within
    Line_In.

    Phrase_In: A string containing the phrase we're searching for.

    Case_Sensitive: Controls if the search is case sensitive or not. (see
    Read_Line_After's docstring for more details).

    ----------------------------------------------------------------------------
    Returns:

    If Phrase_In is a substring of Line_In, then this returns the index of the
    first character after the first instance of Phrase_In within Line_In. If
    Phrase_In is NOT a substring of Line_In, then this function returns -1. """

    # First, get the number of characters in Line/Phrase.
    Num_Chars_Line : int = len(Line_In);
    Num_Chars_Phrase : int = len(Phrase_In);

    # If we're ignoring case, then map Phrase, Line to lower case versions of
    # themselves. Note: We don't want to modify the original variables. Since
    # Python passes by references, we store this result in a copy of Line/Phrase
    Phrase = Phrase_In;
    if(Case_Sensitive == False):
        Line = Line.lower();
        Phrase = Phrase.lower();

    # If Phrase is a substring of Line, then the first character of Phrase must
    # occur before the (Num_Chars_Line - Num_Chars_Phrase) character of Line.
    # Thus, we only need to check the first (Num_Chars_Line - Num_Chars_Phrase)
    # characters of Line.
    for i in range(Num_Chars_Line - Num_Chars_Phrase + 1):
        # Check if ith character of Line_Copy matches the 0th character of
        # Phrase. If so, check if for each j in {0,... Num_Chars_Phrase - 1},
        # Line[i + j] == Phrase[j] (think about it).
        if(Line[i] == Phrase[0]):
            Match : Bool = True;

            for j in range(1, Num_Chars_Phrase):
                # If Line[i + j] != Phrase[j], then we do not have a match, we
                # should move onto the next character of Line.
                if(Line[i + j] != Phrase[j]):
                    Match = False;
                    break;

            # If Match is still True, then Phrase is a substring of Line and
            # i + Num_Chars_Phrase is the index of the first character in Line
            # after Phrase.
            if(Match == True):
                return i + Num_Chars_Phrase;

    # If we're here, then Phrase is NOT a substring of Line. Return -1 to
    # indiciate that.
    return -1;



def Read_Line_After(File, Phrase : str, Case_Sensitive = False) -> str:
    """ This function tries to find a line of File that contains Phrase as a
    substring. Note that we start searching at the current position of the file
    pointer. We do not search from the start of File.

    ----------------------------------------------------------------------------
    Arguments:

    File: The file we want to search for Phrase.

    Phrase: The Phrase we want to find.

    Case_Sensitive: Controls if the search is case sensitive or not. If
    True, then we search for an exact match (including case) of Phrase in one of
    File's lines. If not, then we try to find a line of File which contains the
    same letters in the same order as Phrase.

    ----------------------------------------------------------------------------
    Returns:

    If Phrase is a substring of a line of File, then this function returns
    everything in that line after the first occurrence of Phrase. If it can't
    find Phrase in one of File's lines, it raises an exception. If the Phrase is
    "cat is", and one of File's lines is "the cat is fat", then this will return
    " fat". """

    # Search the lines of File for one that contains Phrase as a substring.
    while(True):
        # Get the next line
        Line = File.readline();

        # Python doesn't use end of file characters. However, readline will
        # retun an empty string if and only if we're at the end of File. Thus,
        # we can use this as our "end of file" check
        if(Line == ""):
            raise Read_Error("Could not find \"" + Phrase + "\" in File.");

        # If the line is a comment (starts with '#'), then ignore it.
        if (Line[0] == "#"):
            continue;

        # Check if Phrase is a substring of Line. If so, this will return the
        # index of the first character after phrase in Line. In this case,
        # return everything in Line after that index. If Phrase is not in
        # Line. In this case, move on to the next line.
        Index : int = Index_After_Phrase(Line, Phrase, Case_Sensitive);
        if(Index == -1):
            continue;
        else:
            return Line[Index:];



def Settings_Reader() -> Settings_Container:
    """ This function reads the settings in Settings.txt.

    ----------------------------------------------------------------------------
    Arguments:

    None!

    ----------------------------------------------------------------------------
    Returns:

    A Settings_Container object that contains all the settings we read from
    Settings.txt. The main function uses these to set up the program. """

    # Open file, initialze a Settings object.
    File = open("../Settings.txt", "r");
    Settings = Settings_Container();



    ############################################################################
    # Save, Load, Plot Settings

    # Load Sol network state from File?
    Buffer = Read_Line_After(File, "Load Sol Network State [bool] :").strip();
    if  (Buffer[0] == 'T' or Buffer[0] == 't'):
        Settings.Load_Sol_Network_State = True;
    elif(Buffer[0] == 'F' or Buffer[0] == 'f'):
        Settings.Load_Sol_Network_State = False;
    else:
        raise Read_Error("\"Load Sol Network State\" should be \"True\" or \"False\". Got " + Buffer);

    # Load PDE network state from File?
    Buffer = Read_Line_After(File, "Load PDE Network State [bool] :").strip();
    if  (Buffer[0] == 'T' or Buffer[0] == 't'):
        Settings.Load_PDE_Network_State = True;
    elif(Buffer[0] == 'F' or Buffer[0] == 'f'):
        Settings.Load_PDE_Network_State = False;
    else:
        raise Read_Error("\"Load PDE Network State\" should be \"True\" or \"False\". Got " + Buffer);

    # Load optimizer state from file?
    Buffer = Read_Line_After(File, "Load Optimizer State [bool] :").strip();
    if  (Buffer[0] == 'T' or Buffer[0] == 't'):
        Settings.Load_Optimize_State = True;
    elif(Buffer[0] == 'F' or Buffer[0] == 'f'):
        Settings.Load_Optimize_State = False;
    else:
        raise Read_Error("\"Load Optimizer State\" should be \"True\" or \"False\". Got " + Buffer);

    # If we are loading anything, get load file name.
    if(     Settings.Load_Sol_Network_State == True or
            Settings.Load_PDE_Network_State == True or
            Settings.Load_Optimize_State  == True):

        Settings.Load_File_Name = Read_Line_After(File, "Load File Name [str] :").strip();

    # Should we plot the final results?
    Buffer = Read_Line_After(File, "Plot Final Result [bool] :").strip();
    if  (Buffer[0] == 'T' or Buffer[0] == 't'):
        Settings.Plot_Final_Results = True;
    elif(Buffer[0] == 'F' or Buffer[0] == 'f'):
        Settings.Plot_Final_Results = False;
    else:
        raise Read_Error("\"Plot Final Result\" should be \"True\" or \"False\". Got " + Buffer);

    # Should we save the network/optimizer state to file?
    Buffer = Read_Line_After(File, "Save State [bool] :").strip();
    if(Buffer[0] == 'T' or Buffer[0] == 't'):
        Settings.Save_To_File = True;
    elif(Buffer[0] == 'F' or Buffer[0] == 'f'):
        Settings.Save_To_File = False;
    else:
        raise Read_Error("\"Save State\" should be \"True\" or \"False\". Got " + Buffer);

    # Should we save the plot?
    Buffer = Read_Line_After(File, "Save Plot [bool] :").strip();
    if(Buffer[0] == 'T' or Buffer[0] == 't'):
        Settings.Save_Plot = True;
    elif(Buffer[0] == 'F' or Buffer[0] == 'f'):
        Settings.Save_Plot = False;
    else:
        raise Read_Error("\"Save Plot\" should be \"True\" or \"False\". Got " + Buffer);

    # If we want to save anything, get save file name.
    if(Settings.Save_To_File == True or Settings.Save_Plot == True):
        Settings.Save_File_Name = Read_Line_After(File, "Save File Name [str] :").strip();




    ############################################################################
    # Mode

    # PINNS, PDE Discovery, or PDE Extraction mode?
    Buffer = Read_Line_After(File, "PINNs, Discovery, or Extraction mode [str] :").strip();
    if  (Buffer[0] == 'P' or Buffer[0] == 'p'):
        Settings.Mode = "PINNs";
    elif(Buffer[0] == 'D' or Buffer[0] == 'd'):
        Settings.Mode = "Discovery";
    elif(Buffer[0] == 'E' or Buffer[0] == 'e'):
        Settings.Mode = "Extraction";
    else:
        raise Read_Error("\"PINNs, Discovery, or Extraction mode\" should be" + \
                         "\"PINNs\", \"Discovery\", or \"Extraction\". Got " + Buffer);

    # PINNs mode specific settings
    if(Settings.Mode == "PINNs"):
        Settings.Periodic_BCs_Highest_Order   = int(Read_Line_After(File, "Periodic BCs highest order [int] :").strip());
        Settings.Num_Train_Colloc_Points = int(Read_Line_After(File, "Number of Training Collocation Points [int] :").strip());
        Settings.Num_Test_Colloc_Points  = int(Read_Line_After(File, "Number of Testing Collocation Points [int] :").strip());

    # Discovery mode specific settings
    if(Settings.Mode == "Discovery"):
        Settings.Num_Train_Data_Points   = int(Read_Line_After(File, "Number of Data Training Points [int] :").strip());
        Settings.Num_Test_Data_Points    = int(Read_Line_After(File, "Number of Data Testing Points [int] :").strip());
        Settings.Num_Train_Colloc_Points = int(Read_Line_After(File, "Number of Training Collocation Points [int] :").strip());
        Settings.Num_Test_Colloc_Points  = int(Read_Line_After(File, "Number of Testing Collocation Points [int] :").strip());

    # Extraction mode specific settings.
    if (Settings.Mode == "Extraction"):
        Settings.Extracted_term_degree   = int(Read_Line_After(File, "Extracted PDE maximum term degree [int] :").strip());
        Settings.Num_Extraction_Points   = int(Read_Line_After(File, "Number of Extraction Points [int] :").strip());

    # Should we try to learn on a GPU?
    Buffer = Read_Line_After(File, "Train on CPU or GPU [GPU, CPU] :").strip();
    if(Buffer[0] == 'G' or Buffer[0] == 'g'):
        if(torch.cuda.is_available() == True):
            Settings.Device = torch.device('cuda');
        else:
            Settings.Device = torch.device('cpu');
            print("You requested a GPU, but cuda is not available on this machine. Switching to CPU");
    elif(Buffer[0] == 'C' or Buffer[0] == 'c'):
        Settings.Device = torch.device('cpu');
    else:
        raise Read_Error("\"Train on CPU or GPU\" should be \"CPU\" or \"GPU\". Got " + Buffer);



    ############################################################################
    # Network Settings

    # Read u's network Architecture
    Settings.Sol_Num_Hidden_Layers   = int(Read_Line_After(File, "Sol Network - Number of Hidden Layers [int] :").strip());
    Settings.Sol_Neurons_Per_Layer   = int(Read_Line_After(File, "Sol Network - Neurons per Hidden Layer [int] :").strip());

    Buffer = Read_Line_After(File, "Sol Network - Activation Function [str] :").strip();
    if  (Buffer[0] == 'R' or Buffer[0] == 'r'):
        Settings.Sol_Activation_Function = "Rational";
    elif(Buffer[0] == 'S' or Buffer[0] == 'S'):
        Settings.Sol_Activation_Function = "Sigmoid";
    elif(Buffer[0] == 'T' or Buffer[0] == 't'):
        Settings.Sol_Activation_Function = "Tanh";
    else:
        raise Read_Error("\"Sol Network - Activation Function [str] :\" should be one of" + \
                         "\"Tanh\", \"Sigmoid\", or \"Rational\" Got " + Buffer);


    # Read N's network Architecture
    Settings.PDE_Num_Sol_derivatives = int(Read_Line_After(File, "PDE Network - PDE Order [int] :").strip());
    Settings.PDE_Num_Hidden_Layers   = int(Read_Line_After(File, "PDE Network - Number of Hidden Layers [int] :").strip());
    Settings.PDE_Neurons_Per_Layer   = int(Read_Line_After(File, "PDE Network - Neurons per Hidden Layer [int] :").strip());


    Buffer = Read_Line_After(File, "PDE Network - Activation Function [str] :").strip();
    if  (Buffer[0] == 'R' or Buffer[0] == 'r'):
        Settings.PDE_Activation_Function = "Rational";
    elif(Buffer[0] == 'S' or Buffer[0] == 'S'):
        Settings.PDE_Activation_Function = "Sigmoid";
    elif(Buffer[0] == 'T' or Buffer[0] == 't'):
        Settings.PDE_Activation_Function = "Tanh";
    else:
        raise Read_Error("\"PDE Network - Activation Function [str] :\" should be one of" + \
                         "\"Tanh\", \"Sigmoid\", or \"Rational\" Got " + Buffer);

    # Read optimizer.
    Buffer = Read_Line_After(File, "Optimizer [Adam, LBFGS] :").strip();
    if  (Buffer[0] == 'A' or Buffer[0] == 'a'):
        Settings.Optimizer = "Adam";
    elif(Buffer[0] == 'L' or Buffer[0] == 'l'):
        Settings.Optimizer = "LBFGS";
    else:
        raise Read_Error("\"Optimizer [Adam, LBFGS] :\" should be \"Adam\" or \"LBFGS\". Got " + Buffer);

    # Read floating point precision
    Buffer = Read_Line_After(File, "FP Precision [Half, Single, Double] :").strip();
    if  (Buffer[0] == 'S' or Buffer[0] == 's'):
        Settings.Torch_dtype = torch.float32;
        Settings.Numpy_dtype = numpy.float32;
    elif(Buffer[0] == 'D' or Buffer[0] == 'd'):
        Settings.Torch_dtype = torch.float64;
        Settings.Numpy_dtype = numpy.float64;
    elif(Buffer[0] == 'H' or Buffer[0] == 'h'):
        Settings.Torch_dtype = torch.float16;
        Settings.Numpy_dtype = numpy.float16;
    else:
        raise Read_Error("\"Floating Point Precision [Single, Double] :\" should be \"Single\" or \"Double\". Got " + Buffer);




    ############################################################################
    # Learning hyper-parameters

    Settings.Epochs = int(Read_Line_After(File, "Number of Epochs [int] :").strip());
    Settings.Learning_Rate = float(Read_Line_After(File, "Learning Rate [float] :").strip());

    if(Settings.Mode == "PINNs" or Settings.Mode == "Discovery"):
        Settings.Epochs_For_New_Coll_Pts = int(Read_Line_After(File, "Epochs between new Collocation Points [int] :").strip());

    Settings.Epochs_Between_Prints = int(Read_Line_After(File, "Epochs between testing [int] :").strip());



    ############################################################################
    # Data

    Settings.Data_File_Name        = Read_Line_After(File, "Data File [str] :").strip();
    Settings.Time_Series_Label     = Read_Line_After(File, "Time Variable Series Label [str] :").strip();
    Settings.Space_Series_Label    = Read_Line_After(File, "Space Variable Series Label [str] :").strip();
    Settings.Solution_Series_Label = Read_Line_After(File, "Solution Series Label [str] :").strip();

    Settings.Noise_Proportion      = float(Read_Line_After(File, "Noise Proportion [float] :").strip());

    # All done! Return the settings!
    File.close();
    return Settings;
