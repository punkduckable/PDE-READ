# Library Structure: #
`PDE-READ` is a library for identifying PDEs from noisy and limited spatiotemporal data. It implements the algorithm described in my paper "PDE-READ: Human-readable Partial Differential Equation Discovery using Deep Learning." `PDE-READ`'s code is in the Code sub-directory. `PDE-READ`has many settings, all of which are in the Settings file (`Settings.txt`). `PDE-READ` has two modes of operation: Discovery and Extraction.

In Discovery mode, `PDE-READ` trains two neural networks, $U$ and $N$. `PDE-READ` trains $U$ to match a data set (specified via the "DataSet Name" setting). Simultaneously, it trains $N$ to satisify
$$D_t^m U = N(U, D_x U, D_x^2 U, \ldots , D_x^n U)$$
You specify $n$ and $m$ in the expression above using the "PDE - Time Derivative Order" and "PDE - Spatial Derivative Order" settings, respectively. Currently, $U$ must be a function of two variables ($t$, and $x$), though we may expand support to more variables in the future. `PDE-READ` often refers to $U$ as the "Solution Network" and $N$ as the "PDE Network".

`PDE-READ`'s Extration mode uses Recursive Feature Elimination (REF), a sparse regression algorithm, to find a sparse setup coefficients $C_1, \ldots , C_N$ in the equation
$$N(U, D_x U, \ldots , D_x^m U) = C_1 F_1(U) + \cdots + C_N F_N(U) \tag{a}$$
Here, each $F_i$ is a function of $U$ and its spatial partial derivatives. In the current implementation,
$$\{ F_1, \ldots , F_N \} = \{ ((U)^{P_0})((D_x U)^{P_1})*\cdots*((D_x^n U)^{P_n} ) : P_0 + \cdots + P_n \leq K \} \tag{b}$$
The "Extracted PDE maximum term degree" setting specifies $K$. RFE uses an iterative process to find a sequence of progressively sparser canaidate solutions to (a). It then ranks the candidate solutions and reports the five highest ranking candidates. See my paper for detials.

*Data:* `PDE-READ` trains U to match a data set. The "DataSet Name" setting specifies the data set that `PDE-READ` trains on. "DataSet Name" should be the name of a file in the `Data/DataSets` directory. A "DataSet" is simply a `.npz` file that contains five variables: Training_Inputs, Training_Targets, Testing_Inputs, Testing_Targets, and Bounds. Each should be a `numpy.ndarray` object. If you want to use `PDE-READ` on your data, you must write a program that calls the `Create_Data_Set` function (in `Create_Data_Set.py`) with the appropriate arguments. See that function's docstring for details. Alternatively, you can create a DataSet using one of our `MATLAB` data sets by running `Python3 ./From_MATLAB.py` when your current working directory is `Data`. The `From_MATLAB` file contains four settings: "Data_File_Name", "Noise_Proportion", "Num_Train_Examples", and "Num_Test_Examples". "Data_File_Name" should refer to one of the `.mat` files in the `MATLAB/Data` directory. "Noise_Proportion", "Num_Train_Examples", and "Num_Test_Examples" control the level of noise in the data, the number of training data points, and the number of testing data points, respectively.

*Plot:* Visualizing the solution that `PDE-READ` identifies can be very useful. The `Plot` directory contains code for visualizing the networks that `PDE-READ` trains. Plotting is controlled by the `Plot/Settings.txt` file. This settings file contains many of the same settings as Settings.txt, and they have the same meaning. Critically, the "Load File Name" setting must refer to a file in `Saves`. Further, the network architecture and PDE Settings in `Plot/Settings.txt` must match the values you used when training the network housed in the load file. For example, if the saved trained the network with a rational neural network with five layers and 50 neurons per layer, then `Plot/Settings.txt` must use those settings. To plot a saved solution, set the appropriate settings in `Plot/Settings.txt` and then run `Python3 ./Plot_Solution.py` when your current working directory is `Plot.`

*Everything else:* The other directories in this repository are as follows: `Figures` houses figured produced by running `Plot_Solution.py.` Saves houses serializations of the networks that `PDE-READ` runs ("Load File Name" in `Settings.txt` and `Plot/Settings.txt` must refer to a file in this directory). `Test` contains test code that I used while developing `PDE-READ`. `MATLAB` contains the `MATLAB` data sets (the `.mat` files in `MATLAB/Data`), and the scripts that created them (the `.m` files in `MATLAB/Scripts`).



# Settings: #
`PDE-READ` is entirely controlled from the settings file (Settings.txt). You should not need to modify any code in the `Code` directory; the settings file controlls everything. I have organized the settings into categories depending on what aspects of `PDE-READ` they control. Below is a discussion of each settings category.

*Save, Load Settings:* The "Save State" setting specifies if `PDE-READ` should save the network and optimizer parameters after training. You should almost always set this setting to true. "Load Sol Network State", "Load PDE Network State", and "Load Optimizer State" specify if you want to start training using a pre-saved Solution Network, PDE Network, or Optimizer state, respectively. If any one of these settings is true, you must specify the "Load File Name" setting, which should specify a file in the `Saves` directory. Note: the load file must be a save produced by a previous run of `PDE-READ`. Further, all loaded parameters must come from the same file (currently, there is no way to load the PDE Network state from one save, and the optimizer state from another save, for example).

*Mode:*. These settings control which mode the code runs in (Discovery vs. Extraction) and how each mode runs. You must specify the `DataSet Name` setting, which specifies the data set that `U` trains on (it also specifies the bounds for the collocation and extraction points, the latter of which we need for extraction mode). The "Number of Training Collocation Points" and "Number of Testing Collocation Points" settings control the number of testing and training collocation points, respectively. Note: `PDE-READ` reselects the training collocation points each epoch and the testing collocation points each time we use the test set. "Extracted PDE maximum term degree" specifies $K$ in equation (b) above. "Number of Extraction Points" specifies the number of extraction points. Note that our paper defines and describes `collocation point` and `extraction point.` Finally, "Train on CPU or GPU" specifies if training should happen on a CPU or GPU. You can only train on a GPU if `torch` supports GPU training on your computer's graphics card. Check torch's website for details.  

*PDE settings:*. These settings characterize the PDE that `PDE-READ` tries to learn. The "PDE - Time Derivative Order" and "PDE - Spatial Derivative Order" settings specify m and n in equation (a), respectively.

*Network Settings:* These settings control the architecture of $U$ (the "Sol Network") and $N$ (the "PDE Network"). You specify both network's number of hidden layers, the number of neurons per hidden layer, and the activation function. Additionally, you can optionally add a batch normalization layer before the first layer of the PDE Network by setting "PDE Network - Normalize Inputs" True. This is an experimental feature that can sometimes help `PDE-READ.` Finally, the "Optimizer" setting specifies which optimizer to train the networks. Critically, if you load the solution network from a save, then the solution network settings (number of hidden layers, neurons per hidden layer, and activation function) MUST match those of the network you are loading. The same applies to the PDE Network. Further, if you are loading the optimizer, then the "Optimizer" setting must match the saved optimizer.

*Learning hyper-parameters:* These settings control how `PDE-READ` trains the networks. "Number of Epochs" and "Learning Rate" specify the number of epochs and the learning rate for the optimizer specified in the "Optimizer" setting, respectively. "Epochs between testing" controls the number of epochs between using the Testing set. Note that testing more frequently will increase runtime.



# Running the code: #
 Once you have selected the appropriate settings, you can run the code by entering the `Code` directory (`cd ./Code`) and running the main file (`Python3 ./main.py`).

 *What to do if you get nan:* `PDE-READ` can use the `LBFGS` optimizer. Unfortunately, PyTorch's `LBFGS` optimizer is known to yield nan (see <https://github.com/pytorch/pytorch/issues/5953>). Using the `LBFGS` optimizer occasionally causes `PDE-READ` to break down and start reporting nan. If this occurs, you should kill `PDE-READ` (in the terminal window, press `Ctrl + C`), and then re-run `PDE-READ.` Since `PDE-READ` randomly samples the collocation points from the problem domain, no two runs of `PDE-READ` are identical. Thus, even if you keep the settings the same, re-running `PDE-READ` may avoid the nan issue. If you encounter nan on several successive runs of `PDE-READ,` reduce the learning rate by a factor of $10$ and try again. If all else fails, consider training using another optimizer.


# Dependencies: #
`PDE-READ` will not run unless you have installed the following:
* `Python3`
* `numpy`
* `torch`

Additionally, you'll need `matplotlib` if you plan to run anything in the `Plot` directory, and `scipy` if you plan to use the `From_MATLAB.py` function in the `Data` directory.
