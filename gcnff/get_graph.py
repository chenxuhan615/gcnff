import os
import json
import pickle
import datetime
from gcnff.graph import getatom_num,generate_graphs1,generate_graphs2

def load_file(config):
    f = open(config)
    return json.load(f)

def get_graph(config_file):
    config = load_file(config_file)
    
    print("\n\t------------------Step--One---------------------\n")
    print("\tCreating a folder 'graphdata' in the current directory \n\t\tto store the graph format of each input file.\n")
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
        file_path = config['get_graph']['file_path']
    except:
        print("\tInput file directory ->['get_graph']['file_path']<- must be specified by yourself!\n")
        exit(1)
    try:
        jumpNum = config['get_graph']['jumpNum']
    except:
        jumpNum = 1
        print("\tYou did not set the value of ->['get_graph']['jumpNum']<-, and it has been set to 1 by default!\n")
    try:
        baseEvalue = config['get_graph']['baseEvalue']
    except:
        baseEvalue = 0.0
        print("\tYou did not set the value of ->['get_graph']['baseEvalue']<-, and it has been set to 0.0 by default!\n")

    graph_path="./graphdata"
    isExists=os.path.exists(graph_path)
    if not isExists:
        os.mkdir(graph_path)
    else:
        print("\t\tgraph path exists!\n")

    file_names = os.listdir(file_path)
    for file in file_names:
        atom_num=getatom_num(file_path+'/'+file)
        time_beg_epoch=datetime.datetime.now()
        try:
            graph_list=list()
            generate_graphs2(file_path+'/'+file,graph_list,CUTOFF_DISTANCE1,CUTOFF_DISTANCE2,jumpNum,baseEvalue)
        except:
            graph_list=list()
            generate_graphs1(file_path+'/'+file,graph_list,CUTOFF_DISTANCE1,CUTOFF_DISTANCE2,jumpNum,baseEvalue)
        with open(graph_path+'/'+file+'.pickle','wb') as f:
            pickle.dump(graph_list,f)
        time_end_epoch=datetime.datetime.now()
        print("file-->",file,"<--saved\tincluding atom:",atom_num,"\ttotal configs:\t",len(graph_list),"\ttime = ",time_end_epoch-time_beg_epoch)
    print("\n\t------------------------------------------------\n")
