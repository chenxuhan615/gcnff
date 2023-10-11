### 1.Intro
> GCNFF is a molecular dynamics (MD) force field based on the graph convolutional network (GCN) method, which realizes the transfer of high-precision first-principles methods to molecular dynamics by the LAMMPS.
### 2.Install GCNFF
> So far, our GCNFF only supports the following package installation. To install GCNFF to lammps, go to file README.txt in the ${GCNFF}/USER-GCNFF directory
#### 2.1 Download
> Please download the package of our source code in Github or Gitee.
> `git clone https://gitee.com/chenxuhan615/gcnff.git`
> `git clone https://github.com/chenxuhan615/gcnff.git`
#### 2.2 Configure virtual environment
> We recommend that you use [anaconda](https://www.anaconda.com) to manage your Python environment:
> `conda create -n GCNFF python=3.8`
> According to your own needs, choose to install the PyTorch and PyTorch geometric packages you want in our ${GCNFF}/setup.py file：
```    
# install pytorch
return_code = os.system('pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html')
# install torch-scatter
return_code = os.system('pip3 install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html')
# install torch-sparse
return_code = os.system('pip3 install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html')
# install torch-cluster
return_code = os.system('pip3 install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html')
# install torch-spline-conv
return_code = os.system('pip3 install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html')
# install torch-geometric
return_code = os.system('pip3 install torch-geometric')
```
> Note: you can find the version you want on the corresponding official website, and please replace the version text in the above setup.py file. For example, the cu111 above meaning cuda 11.1 version can be replaced with cpu.
> In addition, you can also install yourself with the guidelines of the two websites:[PyTorch](https://pytorch.org/) and [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).
#### 2.3 Install
> When you have determined the version of the dependent package in the step 2.2, please execute the following command in ${GCNFF}:
> `python setup.py install`
> Wait patiently for the installation process ... 
> Enter `gcnff` and you will see the help information, which represents your successful installation.
> You can see that: 
> - The first column is a brief description of the GCNFF;
> - The second column shows the style of GCNFF commands;
> - The third column shows 8 optional [action commond]s mentioned in second column;
> - You can use the 8th command showfig to view the files generated during the running of GCNFF.

### 3.Use GCNFF
> The following describes the method of training GCNFF. For use in lammps, please refer to the README.txt under file ${GCNFF}/USER-GCNFF
#### 3.1 Prepare data
> Before training, we need to prepare structure data calculated from the first principles as a specific format：
> You can view the examples in the directory ${GCNFF}/examples/data/
> - There are three numbers in the first line: 1. total atoms number; 2. atoms number in one structure; 3. structure number in this file;
> - The second to fourth lines place the lattice vector of the structure;
> - Next, there are several lines to store the information of each atom, namely: atom type, x coordinate, y coordinate, z coordinate, x force, y force, z force, bader charge(optional); 
> - The next line is the potential energy of structure;
> - The last line is an optional line, you can put the virial information of the structure here, and the order is: v11, v22, v33, v12, v13, v23.
> Note: structures in one file need to have same form (the atoms number remains unchanged, including bader charge or not, and including virial information or not). By default, we assume that the unit of position in your data is angstrom (Å), the unit of energy and virial is electron volt (eV), and the unit of force is electron volt per angstrom (eV/Å).
#### 3.2 action commond
> The following briefly introduces the functions of the six [action commond]:
###### Part one: Data pre-processing
> - `get_graph`: Convert the original structure extracted from first-principles calculations into graph structures. After executing the command, a directory named graphdata will be generated in the current directory to store graph data.
###### Part two: Training
> - `divide_set`: Disrupt the order of all graph structures and divide them into training set, validation set and test set according to your custom ratio. After executing the command, a directory named **model_graphdata** will be generated in the current directory to store the divided graph data.
> - `divide_fix_train`: This initial training will use a small amount of data (for example: a quarter) from your total data to train a model with all convolutional layer parameters fixed. The purpose of this is to reduce the complexity of one-step training. After executing the command, a file named **Epoches-Errors-init.txt** will be generated in the current directory to record the error changes during the training process and a file named **best_model_init.pkl** is the binary format of the initial model.
> - `divide_free_train`: On the basis of the initial training model, use all the data for final training to get the final accurate model. Similar to the previous command, two files with similar naming rules will be generated.
> The above training method usually requires a large amount of computer memory, because all data is scrambled; the following two commands only scramble each input：
> - `direct_fix_train`: The function of this command is similar to command `divide_fix_train`.
> - `direct_free_train`: The function of this command is similar to command `divide_free_train`.
###### Part three: Test
> - `model_test`: You can use this command to test any model with the data you specify. After executing this command, 2 (or 3, if there is virial information) files will be generated. The two columns of data in these files are the predicted value and the true value respectively, just like names of these files.
###### Part Four: Output
> - `get_potential`: With this command, you will output all the parameters of the model you specify. This file named **potential.txt** can be directly read by LAMMPS to run molecular dynamics.
> If you want to experience the one-key generation potential function, you can choose to use a bash script named *sh_gcnff* (in the ${GCNFF}/examples).
> Notice: All displays of GCNFF during execution are saved to the **gcnff.log** file for future reference.
#### 3.3 input file
> The parameter file *input.json* we recommend has been placed in directory: ${GCNFF}/examples/train.
> If you don't know how to choose the value of following keywords, in most cases you can remove this option, because we have basically given default values for all keywords. The most simplified file can be like this *input_sim.json*(in the ${GCNFF}/examples/train)
###### "get_graph"：Scheme used when generating graph structure
> - `CUTOFF_DISTANCE1`: Cut-off radius considered for each atom
> - `CUTOFF_DISTANCE2`: Interconnections between atoms in the selected local environment
> - `file_path`: The directory where the files extracted from the first principles are stored
> - `jumpNum`: Extract a data from input every ${jumpNum} steps, if you want to use all data of input file, please set it to 1
> - `baseEvalue`: In order to avoid outrageous data, the data whose energy value exceeds ${baseEvalue} eV/atom is eliminated.
>    Note: The first and second parameter will also be needed by [action commond] 3 and 4.
###### "divide_set"：Scheme used to split data
> - `RandomSeed`: Random seed to disrupt graph data
> - `graphfile_path`: Graph data storage directory
> - `initmodel_data`: The proportion of data used for initial model training to all data
> - `initmodel_traindata`: The data used for initial model is divided into training set and validation set. Here is the proportion of training set.
> - `finalmodel_traindata`: The training of the final model will use all the data, and divided into: training set, validation set and test set. Here is the proportion of the training set.
> - `finalmodel_validata`: The proportion of the validation set of the final model.
###### "training"：Parameters in initial and final model
> - `GAMMA`, `RBF_KERNEL_NUM`, `HID_DIM`, `NUM_CONV`, `EXPONENT`: Model tuning hyperparameters.
> - `RHO1`: The loss function consists of three parts: energy, force and virial. Here is the proportion of energy. (The proportion of force loss is equal to: 1-`RHO1`)
> - `RHO2`: The proportion of virial in the loss function.
> - `Flag_AutoLoss`: If you donnot know how to choose the proportion of loss above, you can set this option to "True". This effect will cover the above proportion , and a multi-objective optimization method is used.
> - `LEARNING_RATE_INIT`: Initial learning rate.
> - `ATOM_TYPES`: Total atom types contained.
> - `file_path`: The directory to store the divided graph data.
> - `pin_memory`: if the device has large memory capacity, it could be set as True to accelerate the training, validation & test process.
> - `batch_num`: The batch size for mini-batch stochastic gradient descent. But GCNFF only supports setting to 1 currently.
> - `LRStep`, `LRGamma`: The learning rate will be ${LRGamma} times every ${LRStep} epochs.
> - `CNT`: The condition for convergence is that the validation set error increases for ${CNT} consecutive Epochs.
> - `batch_step`: In an Epoch, loss information will be displayed every ${batch_step} batches.
> - `max_epoch`: Maximum epoch number allowed for training.
> - `use_device`: You can choose the equipment, such as "cuda:0" or "cpu".
> - `Data_shuffle`: If set to "True", the data will be reshuffled at every epoch. 
> Note: The RHO1 and RHO2 parameter also be set as a function of Epoch i, for example: "0.1", "0.005*i" or "math.exp(i-100)", and quotation marks are essential.
###### "final_model"：Specify the initial model for the retraining model
> This command can also be used when a task is terminated unexpectedly.
> - `begin_model`: Path and name of the initial model.
###### "direct_fix_train"：Read data strategy to train an initial model
> - `RandomSeed`: Random seed to disrupt graph data
> - `graphfile_path`: Graph data storage directory
> - `initmodel_data`: The proportion of data used for initial model training to each input file
> - `initmodel_traindata`: The data used for initial model is divided into training set and validation set. Here is the proportion of training set.
###### "direct_free_train"：Read data strategy to train final model
> - `RandomSeed`: Random seed to disrupt graph data
> - `graphfile_path`: Graph data storage directory
> - `traindata`: Each input used for model is divided into training set and validation set. Here is the proportion of training set.
> - `begin_model`: Path and name of the initial model.
###### "model_test"：Test the trained model
> - `use_device`: You can choose the equipment, such as "cuda:0" or "cpu".
> - `testdata`: Provide test data.
> - `testmodel`: Provide the model to be tested.
###### "get_potential"：Extract model parameters
> - `out_model`: Provide the model to be extracted.
> - `element_list`: Provide a list of chemical names of the elements in the same order as the element types in the input file.
