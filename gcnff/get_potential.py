import json
import torch
import numpy as np
from gcnff.model import GCNFF_init, GCNFF

def load_file(config):
    f = open(config)
    return json.load(f)

def get_potential(config_file):
    config = load_file(config_file)
    print("\n\t------------------Step--Six---------------------\n")
    try:
        Out_model= config['get_potential']['out_model']
    except:
        print("\tTotal atom types ->['get_potential']['out_model']<- must be specified by yourself!\n")
        exit(1)
    try:
        element_list= config['get_potential']['element_list']
    except:
        print("\tTotal atom types ->['get_potential']['element_list']<- must be specified by yourself!\n")
        exit(1)
    try:
        GAMMA = config['training']['GAMMA']
    except:
        GAMMA = 0.1
        print("\tYou did not set the value of ->['training']['GAMMA']<-, and it has been set to 0.1 by default!\n")
    try:
        RBF_KERNEL_NUM = config['training']['RBF_KERNEL_NUM']
    except:
        RBF_KERNEL_NUM = 128
        print("\tYou did not set the value of ->['training']['RBF_KERNEL_NUM']<-, and it has been set to 128 by default!\n")
    try:
        HID_DIM = config['training']['HID_DIM']
    except:
        HID_DIM = 32
        print("\tYou did not set the value of ->['training']['HID_DIM']<-, and it has been set to 32 by default!\n")
    try:
        NUM_CONV = config['training']['NUM_CONV']
    except:
        NUM_CONV = 3
        print("\tYou did not set the value of ->['training']['NUM_CONV']<-, and it has been set to 3 by default!\n")
    try:
        ATOM_TYPES = config['training']['ATOM_TYPES']
    except:
        print("\tTotal atom types ->['training']['ATOM_TYPES']<- must be specified by yourself!\n")
        exit(1)
    try:
        CUTOFF_DISTANCE1 = config['get_graph']['CUTOFF_DISTANCE1']
    except:
        CUTOFF_DISTANCE1 = 6.5
        print("\tYou did not set the value of ->['get_graph']['CUTOFF_DISTANCE1']<-, and it has been set to 6.5 by default!\n")
    try:
        CUTOFF_DISTANCE2 = config['get_graph']['CUTOFF_DISTANCE2']
    except:
        CUTOFF_DISTANCE2 = 5.0
        print("\tYou did not set the value of ->['get_graph']['CUTOFF_DISTANCE2']<-, and it has been set to 5.0 by default!\n")
    try:
        EXPONENT = config['training']['EXPONENT']
    except:
        EXPONENT = 6.0
        print("\tYou did not set the value of ->['training']['EXPONENT']<-, and it has been set to 6.0 by default!\n")
    print("\tGenerating the -> ",Out_model," <- parameters for LAMMPS into -> potential.txt <- .\n")
    out_model = torch.load(Out_model, map_location='cpu')
    filename = 'potential.txt'
    with open(filename, 'w') as f:
        f.write("n_elements %d\n" % ATOM_TYPES)
        f.write("element %s\n" %element_list)
        f.write("HID_DIM %d \n" % HID_DIM)
        f.write("RBF_KERNEL_NUM %d\n" % RBF_KERNEL_NUM)
        f.write("NUM_CONV %d\n" % NUM_CONV)
        f.write("GAMMA %lf\n" % GAMMA)
        f.write("CUTOFF1 %lf\n" %CUTOFF_DISTANCE1)
        f.write("CUTOFF2 %lf\n" %CUTOFF_DISTANCE2)
        f.write("EXPONENT %lf\n" %EXPONENT)
        f.write("Finish\n\n")
        for param in out_model.named_parameters():
            temp_name = param[0]
            print(param[1].size())
            temp = param[1].detach().numpy().reshape(1, -1)
            print(temp_name)
            print(temp.shape)
            # print(temp)
            f.write(temp_name)
            f.write(" ")
            np.savetxt(f, temp)
            f.write("\n")
        element_list=element_list.split(" ")
        element_num=0
        for element in element_list:
            f.write("interaction %s\n" %element)
            f.write("znum %d\n" %element_num)
            f.write("Rc %lf\n" % CUTOFF_DISTANCE1)
            f.write("endVar\n\n")
            element_num+=1
    print("\n\t------------------------------------------------\n")
