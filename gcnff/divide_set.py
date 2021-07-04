import os
import json
import random
import pickle

def load_file(config):
    f = open(config)
    return json.load(f)
    
def divide_set(config_file):
    config = load_file(config_file)
    
    print("\n\t------------------Step--Two---------------------\n")
    print("\tCreating a folder 'model_graphdata' in the current directory \n\t\tto store the divided data set.\n")
    
    try:
        randomSeed=config['divide_set']['RandomSeed']
    except:
        randomSeed=0
        print("\tYou did not set the value of ->['divide_set']['RandomSeed']<-, and it has been set to 0 by default!\n")
    try:
        file_path = config['divide_set']['graphfile_path']
    except:
        file_path = "./graphdata"
        print("\tYou did not set the value of ->['divide_set']['graphfile_path']<-, and it has been set to './graphdata/' by default!\n")
    try:
        date_for_initmodel=config['divide_set']['initmodel_data']
    except:
        date_for_initmodel=0.25
        print("\tYou did not set the value of ->['divide_set']['initmodel_data']<-, and it has been set to 0.25 by default!\n")
    try:
        traindate_for_initmodel=config['divide_set']['initmodel_traindata']
    except:
        traindate_for_initmodel=0.75
        print("\tYou did not set the value of ->['divide_set']['initmodel_traindata']<-, and it has been set to 0.75 by default!\n")
    try:
        traindate_for_finalmodel=config['divide_set']['finalmodel_traindata']
    except:
        traindate_for_finalmodel=0.7
        print("\tYou did not set the value of ->['divide_set']['finalmodel_traindata']<-, and it has been set to 0.7 by default!\n")
    try:
        validate_for_finalmodel=config['divide_set']['finalmodel_validata']
    except:
        validate_for_finalmodel=0.2
        print("\tYou did not set the value of ->['divide_set']['finalmodel_validata']<-, and it has been set to 0.2 by default!\n")

    modelgraph_path="./model_graphdata"
    isExists=os.path.exists(modelgraph_path)
    if not isExists:
        os.mkdir(modelgraph_path)
    else:
        print("\t\tmodel graph path exists!\n")

    file_names=os.listdir(file_path)
    graph_list=list()
    for file in file_names:
        #print(file,"\tread end")
        graph_list+=pickle.load(open(file_path+'/'+file,'rb'))
    print("\tThe total number of graph structures read is:",len(graph_list),".\n")
    
    random.seed(randomSeed)
    random.shuffle(graph_list)
    
    N_init_data = int(date_for_initmodel * len(graph_list))
    N_init_training = int(traindate_for_initmodel * N_init_data)
    print("\n\tFor the init model the training and validation data are:")
    print("\ttraining:\t",len(graph_list[:N_init_training]))
    print("\tvalidation:\t",len(graph_list[N_init_training:N_init_data]))
    
    N_training = int(traindate_for_finalmodel * len(graph_list))
    N_validation = int(validate_for_finalmodel * len(graph_list))
    print("\n\tFor the final model the training, validation and test data are:")
    print("\ttraining:\t",len(graph_list[:N_training]))
    print("\tvalidation:\t",len(graph_list[N_training:(N_training+N_validation)]))
    print("\ttest:\t\t",len(graph_list[(N_training+N_validation):]))
    
    with open(modelgraph_path+'/'+'init_training_graphs.pickle', 'wb') as f:
        pickle.dump(graph_list[:N_init_training], f)
    with open(modelgraph_path+'/'+'init_validation_graphs.pickle', 'wb') as f:
        pickle.dump(graph_list[N_init_training:N_init_data], f)
    with open(modelgraph_path+'/'+'training_graphs.pickle', 'wb') as f:
        pickle.dump(graph_list[:N_training], f)
    with open(modelgraph_path+'/'+'validation_graphs.pickle', 'wb') as f:
        pickle.dump(graph_list[N_training:(N_training + N_validation)], f)
    with open(modelgraph_path+'/'+'test_graphs.pickle', 'wb') as f:
        pickle.dump(graph_list[(N_training + N_validation):], f)
    print("\n\t------------------------------------------------\n")
    