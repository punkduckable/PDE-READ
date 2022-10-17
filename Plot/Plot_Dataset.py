import  numpy;
import  scipy.io;
import  matplotlib.pyplot as plt;


from    Settings_Reader import Settings_Reader, Settings_Container;


class Data_Container:
    def __init__(   self,
                    t_points        : numpy.array,
                    x_points        : numpy.array,
                    Data_Set        : numpy.array,
                    Noisy_Data_Set  : numpy.array, 
                    Data_Set_Name   : str):

        self.t_points       = t_points;
        self.x_points       = x_points;
        self.Data_Set       = Data_Set;
        self.Noisy_Data_Set = Noisy_Data_Set
        self.Data_Set_Name  = Data_Set_Name;



def Load_Dataset(
        Data_Set_File_Name  : str,
        Noise_Level         : float) -> Data_Container:
    """ This function loads the dataset and adds noise to it. We return a
    four element tuple, whose first element contains the raw dataset, and whose
    second element contains the dataset with added noise. """

    # Load data file.
    Data_Set_File_Path  = "../MATLAB/Data/" + Data_Set_File_Name;
    Data_In             = scipy.io.loadmat(Data_Set_File_Path);

    # Fetch spatial, temporal coordinates and the true solution. We cast these
    # to singles (32 bit fp) since that's what the rest of the code uses.
    t_points : numpy.array = Data_In['t'].reshape(-1).astype(dtype  = numpy.float32);
    x_points : numpy.array = Data_In['x'].reshape(-1).astype(dtype  = numpy.float32);
    Data_Set : numpy.array = (numpy.real(Data_In['usol'])).astype(dtype  = numpy.float32);

    # Add noise to true solution.
    Noisy_Data_Set : numpy.array = Data_Set + Noise_Level*(numpy.std(Data_Set)*numpy.random.randn(*Data_Set.shape));

    # Now, make a Data_Container object, package it, and return.
    Container = Data_Container( t_points        = t_points,
                                x_points        = x_points,
                                Data_Set        = Data_Set,
                                Noisy_Data_Set  = Noisy_Data_Set,
                                Data_Set_Name   = Data_Set_File_Name.split('.')[0]);

    return Container;



def Plot_Dataset(       Settings    : Settings_Container,
                        Data        : Data_Container):
    """ This function plots a dataset. """

    # First, set up the figure object.
    plt.style.use('dark_background')
    fig = plt.figure();

    # Next, construct the set of possible coordinates. grid_t_coords and
    # grid_x_coords are 2d NumPy arrays. Each row of these arrays corresponds to
    # a specific position. Each column corresponds to a specific time.
    grid_t_coords, grid_x_coords = numpy.meshgrid(Data.t_points, Data.x_points);

    # Set up the Axes object.
    Axes = fig.add_subplot(1, 1, 1);
    if(Settings.Include_Title == True):
        Axes.set_title(Settings.Plot_Title);
    Axes.set_xlabel("time (s)");
    Axes.set_ylabel("position (m)");

    # Set up the colorbar.
    min : float = numpy.min(Data.Noisy_Data_Set);
    max : float = numpy.max(Data.Noisy_Data_Set);

    ColorMap = Axes.contourf(
                    grid_t_coords,                          # x coordinates in plots
                    grid_x_coords,                          # y coordinates in plot
                    Data.Noisy_Data_Set,                    # z coordinates in plot
                    levels = numpy.linspace(min, max, 500), # defines how z values are mapped to colors.
                    cmap = plt.cm.jet);                     # defines color scheme
    fig.colorbar(ColorMap, ax = Axes, fraction=0.046, pad=0.04, orientation='vertical');

    # Save, show the plot!
    plt.savefig("../Figures/" + Data.Data_Set_Name + "_Dataset.png", dpi = 500, transparent = True);
    plt.show();


if __name__ == "__main__":
    # First, read the settings.
    (_, Settings) = Settings_Reader();

    # Now, load the dataset.
    Data_Container = Load_Dataset(
                        Data_Set_File_Name = Settings.Data_Set_File_Name,
                        Noise_Level        = Settings.Noise_Level);

    # Finally, make the plot!
    Plot_Dataset(Settings   = Settings,
                 Data       = Data_Container);
