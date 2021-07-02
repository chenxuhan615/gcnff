import os
import json
import sys
import random
import pickle
import torch
import datetime
import numpy as np
from torch_geometric.data import DataLoader
from gcnff.graph import getatom_num,generate_graphs1,generate_graphs2
from gcnff.model import GCNFF_init, global_mean_pool, GCNFF
from gcnff.printsave import Logger
from gcnff.showplot import showfig
from gcnff.helpinfo import showhelp
from gcnff.AutomaticWeightedLoss import AutomaticWeightedLoss

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
            generate_graphs1(file_path+'/'+file,graph_list,CUTOFF_DISTANCE1,CUTOFF_DISTANCE2,jumpNum,baseEvalue)
        except:
            graph_list=list()
            generate_graphs2(file_path+'/'+file,graph_list,CUTOFF_DISTANCE1,CUTOFF_DISTANCE2,jumpNum,baseEvalue)
        with open(graph_path+'/'+file+'.pickle','wb') as f:
            pickle.dump(graph_list,f)
        time_end_epoch=datetime.datetime.now()
        print("file-->",file,"<--saved\tincluding atom:",atom_num,"\ttotal configs:\t",len(graph_list),"\ttime = ",time_end_epoch-time_beg_epoch)
    print("\n\t------------------------------------------------\n")

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
    
def init_train(config_file):
    config = load_file(config_file)
    print("\n\t------------------Step--Three---------------------\n")
    print("\tTraining an 'initial model' with fixed parameters in Conv layers.\n")
    try:
        RHO1 = config['training']['RHO1']
    except:
        RHO1 = 0.01
        print("\tYou did not set the value of ->['training']['RHO1']<-, and it has been set to 0.01 by default!\n")
    try:
        RHO2 = config['training']['RHO2']
    except:
        RHO2 = 0.001
        print("\tYou did not set the value of ->['training']['RHO2']<-, and it has been set to 0.001 by default!\n")
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
        LEARNING_RATE_INIT = config['training']['LEARNING_RATE_INIT']
    except:
        LEARNING_RATE_INIT = 0.0002
        print("\tYou did not set the value of ->['training']['LEARNING_RATE_INIT']<-, and it has been set to 0.0002 by default!\n")
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
    try:
        file_path = config['training']['file_path']
    except:
        file_path = "./model_graphdata"
        print("\tYou did not set the value of ->['training']['file_path']<-, and it has been set to './model_graphdata' by default!\n")
    try:
        use_device = config['training']['use_device']
    except:
        use_device= 'cuda:0' if torch.cuda.is_available() else 'cpu' 
        print("\tYou did not set the value of ->['training']['use_device']<-, and it has been set to ",use_device," by default!\n")
    try:
        is_pin_memory = config['training']['pin_memory']
    except:
        is_pin_memory="False"
        print("\tYou did not set the value of ->['training']['pin_memory']<-, and it has been set to 'False' by default!\n")
    try:
        batch_num = config['training']['batch_num']
        if(batch_num!=1):
            print("\tGCNFF only supports the setting of ['training']['batch_num'] to 1 currently\n")
            exit(1)
    except:
        batch_num=1
        print("\t->['training']['batch_num']<- has been set to 1 by default!\n")
    try:
        LRStep = config['training']['LRStep']
    except:
        LRStep =5
        print("\tYou did not set the value of ->['training']['LRStep']<-, and it has been set to 5 by default!\n")
    try:
        LRGamma = config['training']['LRGamma']
    except:
        LRGamma =0.8
        print("\tYou did not set the value of ->['training']['LRGamma']<-, and it has been set to 0.8 by default!\n")
    try:
        CNT = config['training']['CNT']
    except:
        CNT = 10
        print("\tYou did not set the value of ->['training']['CNT']<-, and it has been set to 10 by default!\n")
    try:
        batch_step = config['training']['batch_step']
    except:
        batch_step = 1000
        print("\tYou did not set the value of ->['training']['batch_step']<-, and it has been set to 1000 by default!\n")
    try:
        max_epoch = config['training']['max_epoch']
    except:
        max_epoch = 300
        print("\tYou did not set the value of ->['training']['max_epoch']<-, and it has been set to 300 by default!\n")
    try:
        Data_shuffle = config['training']['Data_shuffle']
    except:
        Data_shuffle ="False"
        print("\tYou did not set the value of ->['training']['Data_shuffle']<-, and it has been set to 'False' by default!\n")
    try:
        Flag_AutomaticWeightedLoss = config['training']['Flag_AutoLoss']
    except:
        Flag_AutomaticWeightedLoss ="False"
        print("\tYou did not set the value of ->['training']['Flag_AutoLoss']<-, and it has been set to 'False' by default!\n")
    
    if (Flag_AutomaticWeightedLoss=="True"):
        awl = AutomaticWeightedLoss(3)
        awl.to(use_device)
        print("\tUsing the AutomaticWeightedLoss method! ->https://github.com/Mikoto10032/AutomaticWeightedLoss<-\n")
    init_training_graph = pickle.load(open(file_path+'/'+'init_training_graphs.pickle', 'rb'))
    init_validation_graph = pickle.load(open(file_path+'/'+'init_validation_graphs.pickle', 'rb'))

    init_train_size = len(init_training_graph)
    init_valid_size = len(init_validation_graph)
    print("\tthe init model has training data:\t",init_train_size)
    print("\tthe init model has validation data:\t",init_valid_size)
    print("\n\n\t\tstarting to training : \n\n")
    if(is_pin_memory=="True"):
        if(Data_shuffle=="True"):
            init_train_dataloader = DataLoader(init_training_graph, batch_size=batch_num, pin_memory=True,shuffle=True)
            init_valid_dataloader = DataLoader(init_validation_graph, batch_size=batch_num, pin_memory=True,shuffle=True)
        else:
            init_train_dataloader = DataLoader(init_training_graph, batch_size=batch_num, pin_memory=True,shuffle=False)
            init_valid_dataloader = DataLoader(init_validation_graph, batch_size=batch_num, pin_memory=True,shuffle=False)
    else:
        if(Data_shuffle=="True"):
            init_train_dataloader = DataLoader(init_training_graph, batch_size=batch_num, pin_memory=False,shuffle=True)
            init_valid_dataloader = DataLoader(init_validation_graph, batch_size=batch_num, pin_memory=False,shuffle=True)
        else:
            init_train_dataloader = DataLoader(init_training_graph, batch_size=batch_num, pin_memory=False,shuffle=False)
            init_valid_dataloader = DataLoader(init_validation_graph, batch_size=batch_num, pin_memory=False,shuffle=False)
    model_init=GCNFF_init(cutoff1=CUTOFF_DISTANCE1,cutoff2=CUTOFF_DISTANCE2,gamma=GAMMA,rbfkernel_number=RBF_KERNEL_NUM,
                            hidden_layer_dimensions=HID_DIM,num_conv=NUM_CONV,atom_types=ATOM_TYPES,exponent=EXPONENT)

    model_init.to(use_device)
    try:
        optimizer_init = torch.optim.Adam([{'params':model_init.parameters()},{'params':awl.parameters(),'weight_decay':0}], lr=LEARNING_RATE_INIT)
    except:
        optimizer_init = torch.optim.Adam(model_init.parameters(), lr=LEARNING_RATE_INIT)
    scheduler_init = torch.optim.lr_scheduler.StepLR(optimizer_init, step_size=LRStep, gamma=LRGamma)

    min_valid_error=np.inf
    train_errors_init=list()
    valid_errors_init=list()
    cnt=0
    for i in range(max_epoch):
        time_beg_epoch=datetime.datetime.now()
        #training process
        try:
            awl.train()
        except:
            pass
        model_init.train()
        train_error=0
        batch_num=0
        for train_graph in init_train_dataloader:
            batch_num+=1
            train_graph=train_graph.to(use_device)
            optimizer_init.zero_grad()
            dist1=torch.index_select(train_graph.pos,0,train_graph.edge_index1[1])-torch.index_select(train_graph.pos,0,train_graph.edge_index1[0])
            dist1=torch.add(dist1,train_graph.edge_attr1)
            try:
                dist2=torch.index_select(train_graph.pos,0,train_graph.edge_index2[1])-torch.index_select(train_graph.pos,0,train_graph.edge_index2[0])
                dist2=torch.add(dist2,train_graph.edge_attr2)
            except:
                dist2=torch.tensor([[],[],[]]).t().to(use_device)
            
            try:
                pred_energy=model_init(train_graph,dist1,dist2,use_device)*100
                pred_force=-torch.autograd.grad(pred_energy,train_graph.pos,
                                                retain_graph=True,create_graph=True,
                                                grad_outputs=torch.ones_like(pred_energy)*100)[0]
                try:
                    true_virial=train_graph.virial
                    pred_fij=-torch.autograd.grad(pred_energy,dist1,
                                                retain_graph=True,create_graph=True,
                                                grad_outputs=torch.ones_like(pred_energy)*100)[0]
                    pred_fjk=-torch.autograd.grad(pred_energy,dist2,
                                                retain_graph=True,create_graph=True,
                                                grad_outputs=torch.ones_like(pred_energy)*100)[0]
                    dist=torch.cat([dist1,dist2]).t()
                    fijk=torch.cat([pred_fij,pred_fjk]).t() 
                    pred_v11=torch.sum(dist[0]*fijk[0]).unsqueeze(0).unsqueeze(0)
                    pred_v22=torch.sum(dist[1]*fijk[1]).unsqueeze(0).unsqueeze(0)
                    pred_v33=torch.sum(dist[2]*fijk[2]).unsqueeze(0).unsqueeze(0)
                    pred_v12=0.5*(torch.sum(dist[0]*fijk[1])+torch.sum(dist[1]*fijk[0])).unsqueeze(0).unsqueeze(0)
                    pred_v13=0.5*(torch.sum(dist[0]*fijk[2])+torch.sum(dist[2]*fijk[0])).unsqueeze(0).unsqueeze(0)
                    pred_v23=0.5*(torch.sum(dist[1]*fijk[2])+torch.sum(dist[2]*fijk[1])).unsqueeze(0).unsqueeze(0)
                    pred_v=torch.cat([pred_v11,pred_v22,pred_v33,pred_v12,pred_v13,pred_v23],1)
                    loss_virial=torch.nn.functional.mse_loss(pred_v,true_virial,reduction='none')
                    loss_virial=torch.sum(loss_virial,dim=(0,1),keepdim=True)/6
                    loss_force=torch.norm(train_graph.force-pred_force,p=2,dim=1,keepdim=True)
                    loss_force=torch.mul(loss_force,loss_force)
                    loss_energy=torch.nn.functional.mse_loss(pred_energy,train_graph.y,reduction='none')
                    loss_force_avg=global_mean_pool(loss_force,train_graph.batch)
                    try:
                        total_loss=awl(loss_energy,loss_force_avg,loss_virial)
                    except:
                        total_loss=RHO1*loss_energy+(1-RHO1)*loss_force_avg+RHO2*loss_virial
                    total_loss.backward(torch.ones_like(total_loss))
                    optimizer_init.step()
                    train_error+=torch.sum(total_loss).cpu().detach().numpy()
                    if batch_num%batch_step==0:
                        print('   batch ',batch_num,' training error = %.2f'%(total_loss.item()),
                                                    ' loss of energy = %.2f'%(loss_energy.item()),
                                                    ' loss of force = %.2f'%(loss_force_avg.item()),
                                                    'loss of virial = %.2f'%(loss_virial.item()))
                except:
                    loss_force=torch.norm(train_graph.force-pred_force,p=2,dim=1,keepdim=True)
                    loss_force=torch.mul(loss_force,loss_force)
                    loss_energy=torch.nn.functional.mse_loss(pred_energy,train_graph.y,reduction='none')
                    loss_force_avg=global_mean_pool(loss_force,train_graph.batch)
                    try:
                        total_loss=awl(loss_energy,loss_force_avg)
                    except:
                        total_loss=RHO1*loss_energy+(1-RHO1)*loss_force_avg
                    total_loss.backward(torch.ones_like(total_loss))
                    optimizer_init.step()
                    train_error+=torch.sum(total_loss).cpu().detach().numpy()
                    if batch_num%batch_step==0:
                        print('   batch ',batch_num,' training error = ',total_loss.item(),
                                                    ' loss of energy = ',loss_energy.item(),
                                                    ' loss of force = ',loss_force_avg.item())
            except:
                torch.cuda.empty_cache()
                print("\tThis training data has  ",train_graph.x.size(0),"\tatoms! May out of memory!")
        train_errors_init.append(train_error/init_train_size)
    
        #validation process
        try:
            awl.eval()
        except:
            pass
        model_init.eval()
        optimizer_init.zero_grad()
        torch.cuda.empty_cache()
        valid_error=0
        for valid_graph in init_valid_dataloader:
            valid_graph=valid_graph.to(use_device)
            dist1=torch.index_select(valid_graph.pos,0,valid_graph.edge_index1[1])-torch.index_select(valid_graph.pos,0,valid_graph.edge_index1[0])
            dist1=torch.add(dist1,valid_graph.edge_attr1)
            try:
                dist2=torch.index_select(valid_graph.pos,0,valid_graph.edge_index2[1])-torch.index_select(valid_graph.pos,0,valid_graph.edge_index2[0])
                dist2=torch.add(dist2,valid_graph.edge_attr2)
            except:
                dist2=torch.tensor([[],[],[]]).t().to(use_device)
            try:
                pred_energy=model_init(valid_graph,dist1,dist2,use_device)*100
                pred_force=-torch.autograd.grad(pred_energy,valid_graph.pos,
                                                retain_graph=True,create_graph=True,
                                                grad_outputs=torch.ones_like(pred_energy)*100)[0]
                try:
                    true_virial=valid_graph.virial
                    pred_fij=-torch.autograd.grad(pred_energy,dist1,
                                                retain_graph=True,create_graph=True,
                                                grad_outputs=torch.ones_like(pred_energy)*100)[0]
                    pred_fjk=-torch.autograd.grad(pred_energy,dist2,
                                                retain_graph=True,create_graph=True,
                                                grad_outputs=torch.ones_like(pred_energy)*100)[0]
                    dist=torch.cat([dist1,dist2]).t()
                    fijk=torch.cat([pred_fij,pred_fjk]).t()
                    pred_v11=torch.sum(dist[0]*fijk[0]).unsqueeze(0).unsqueeze(0)
                    pred_v22=torch.sum(dist[1]*fijk[1]).unsqueeze(0).unsqueeze(0)
                    pred_v33=torch.sum(dist[2]*fijk[2]).unsqueeze(0).unsqueeze(0)
                    pred_v12=0.5*(torch.sum(dist[0]*fijk[1])+torch.sum(dist[1]*fijk[0])).unsqueeze(0).unsqueeze(0)
                    pred_v13=0.5*(torch.sum(dist[0]*fijk[2])+torch.sum(dist[2]*fijk[0])).unsqueeze(0).unsqueeze(0)
                    pred_v23=0.5*(torch.sum(dist[1]*fijk[2])+torch.sum(dist[2]*fijk[1])).unsqueeze(0).unsqueeze(0)
                    pred_v=torch.cat([pred_v11,pred_v22,pred_v33,pred_v12,pred_v13,pred_v23],1)
                    loss_virial=torch.nn.functional.mse_loss(pred_v,true_virial,reduction='none')
                    loss_virial=torch.sum(loss_virial,dim=(0,1),keepdim=True)/6
                    loss_force=torch.norm(valid_graph.force-pred_force,p=2,dim=1,keepdim=True)
                    loss_force=torch.mul(loss_force,loss_force)
                    loss_energy=torch.nn.functional.mse_loss(pred_energy,valid_graph.y,reduction='none')
                    loss_force_avg=global_mean_pool(loss_force,valid_graph.batch)
                    try:
                        total_loss=awl(loss_energy,loss_force_avg,loss_virial)
                    except:
                        total_loss=RHO1*loss_energy+(1-RHO1)*loss_force_avg+RHO2*loss_virial
                    valid_error+=torch.sum(total_loss).cpu().detach().numpy()
                except:
                    loss_energy=torch.nn.functional.mse_loss(pred_energy,valid_graph.y,reduction='none')
                    loss_force=torch.norm(valid_graph.force-pred_force,p=2,dim=1,keepdim=True)
                    loss_force=torch.mul(loss_force,loss_force)
                    loss_force_avg=global_mean_pool(loss_force,valid_graph.batch)
                    try:
                        total_loss=awl(loss_energy,loss_force_avg)
                    except:
                        total_loss=RHO1*loss_energy+(1-RHO1)*loss_force_avg
                    valid_error+=torch.sum(total_loss).cpu().detach().numpy()
            except:
                torch.cuda.empty_cache()
                print("\tThis validataion data has  ",valid_graph.x.size(0),"\tatoms! May out of memory!")
        valid_errors_init.append(valid_error/init_valid_size)
    
        #print information & judgement for early stopping
        scheduler_init.step()
        time_end_epoch=datetime.datetime.now()
        print('Epoch ',i,' training error = ',train_error/init_train_size,
            ' validation error = ',valid_error/init_valid_size,
            ' training and validation time = ',time_end_epoch-time_beg_epoch,
            ' learning rate = ',optimizer_init.param_groups[0]["lr"])

        with open('Epoches-Errors-init.txt', 'w') as f:
            f.write('Epoches\t\tTraining_Error\t\tValidation_Error\n')
            for index in range(len(train_errors_init)):
                f.write("%d\t\t%lf\t\t%lf\n" % (index, train_errors_init[index], valid_errors_init[index]))

        if valid_error<min_valid_error: #judgement for early stopping
            cnt=0
            try:
                torch.save(awl,'AutomaticWeightedLoss_init.pkl')
            except:
                pass
            torch.save(model_init,'best_model_init.pkl')
            min_valid_error=valid_error
        else:
            cnt+=1
            if cnt>=CNT:
                print('Early stopping')
                try:
                    del(awl)
                except:
                    pass
                del(model_init)
                #with open('training_errors_init.pickle','wb') as f:
                #    pickle.dump(train_errors_init,f)
                #with open('valid_errors_init.pickle','wb') as f:
                #    pickle.dump(valid_errors_init,f)
                break
    print("\n\t------------------------------------------------\n")


def final_train(config_file):
    config = load_file(config_file)
    
    print("\n\t------------------Step--Four---------------------\n")
    print("\tRetraining an 'final model' with unfixed parameters in Conv layers.\n")
    try:
        RHO1 = config['training']['RHO1']
    except:
        RHO1 = 0.01
        print("\tYou did not set the value of ->['training']['RHO1']<-, and it has been set to 0.01 by default!\n")
    try:
        RHO2 = config['training']['RHO2']
    except:
        RHO2 = 0.001
        print("\tYou did not set the value of ->['training']['RHO2']<-, and it has been set to 0.001 by default!\n")
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
        LEARNING_RATE_INIT = config['training']['LEARNING_RATE_INIT']
    except:
        LEARNING_RATE_INIT = 0.0002
        print("\tYou did not set the value of ->['training']['LEARNING_RATE_INIT']<-, and it has been set to 0.0002 by default!\n")
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
    try:
        file_path = config['training']['file_path']
    except:
        file_path = "./model_graphdata"
        print("\tYou did not set the value of ->['training']['file_path']<-, and it has been set to './model_graphdata' by default!\n")
    try:
        use_device = config['training']['use_device']
    except:
        use_device= 'cuda:0' if torch.cuda.is_available() else 'cpu' 
        print("\tYou did not set the value of ->['training']['use_device']<-, and it has been set to ",use_device," by default!\n")
    try:
        is_pin_memory = config['training']['pin_memory']
    except:
        is_pin_memory="False"
        print("\tYou did not set the value of ->['training']['pin_memory']<-, and it has been set to 'False' by default!\n")
    try:
        batch_num = config['training']['batch_num']
        if(batch_num!=1):
            print("\tGCNFF only supports the setting of ['training']['batch_num'] to 1 currently\n")
        batch_num=1
    except:
        batch_num=1
        print("\t->['training']['batch_num']<- has been set to 1 by default!\n")
    try:
        LRStep = config['training']['LRStep']
    except:
        LRStep =5
        print("\tYou did not set the value of ->['training']['LRStep']<-, and it has been set to 5 by default!\n")
    try:
        LRGamma = config['training']['LRGamma']
    except:
        LRGamma =0.8
        print("\tYou did not set the value of ->['training']['LRGamma']<-, and it has been set to 0.8 by default!\n")
    try:
        CNT = config['training']['CNT']
    except:
        CNT = 10
        print("\tYou did not set the value of ->['training']['CNT']<-, and it has been set to 10 by default!\n")
    try:
        batch_step = config['training']['batch_step']
    except:
        batch_step = 1000
        print("\tYou did not set the value of ->['training']['batch_step']<-, and it has been set to 1000 by default!\n")
    try:
        max_epoch = config['training']['max_epoch']
    except:
        max_epoch = 300
        print("\tYou did not set the value of ->['training']['max_epoch']<-, and it has been set to 300 by default!\n")
    try:
        Data_shuffle = config['training']['Data_shuffle']
    except:
        Data_shuffle ="False"
        print("\tYou did not set the value of ->['training']['Data_shuffle']<-, and it has been set to 'False' by default!\n")
    try:
        Flag_AutomaticWeightedLoss = config['training']['Flag_AutoLoss']
    except:
        Flag_AutomaticWeightedLoss ="False"
        print("\tYou did not set the value of ->['training']['Flag_AutoLoss']<-, and it has been set to 'False' by default!\n")
    try:
        init_model=config['final_model']['begin_model']
    except:
        print("\tYou did not set the value of ->['final_model']['begin_model']<- !\n")
        
    if (Flag_AutomaticWeightedLoss=="True"):
        awl = AutomaticWeightedLoss(3)
        try:
            awl_init=torch.load("./AutomaticWeightedLoss.pkl",map_location='cpu')
            init_awl_state_dict=awl_init.state_dict()
            awl.load_state_dict(init_awl_state_dict)
            print("\tReading ->AutomaticWeightedLoss.pkl<- success!\n")
        except:
            print("\tCannot reading ->AutomaticWeightedLoss.pkl<- , so we init weight of AutomaticWeightedLoss as ones\n")
        awl.to(use_device)
        print("\tUsing the AutomaticWeightedLoss method! ->https://github.com/Mikoto10032/AutomaticWeightedLoss<-\n")
        
    model_finetune=GCNFF(cutoff1=CUTOFF_DISTANCE1,cutoff2=CUTOFF_DISTANCE2,gamma=GAMMA,rbfkernel_number=RBF_KERNEL_NUM,
                     hidden_layer_dimensions=HID_DIM,num_conv=NUM_CONV,atom_types=ATOM_TYPES,exponent=EXPONENT)
    try:
        model_init=torch.load(init_model,map_location='cpu')
        init_state_dict=model_init.state_dict()
        finetune_state_dict=model_finetune.state_dict()
        for layer_names in init_state_dict:
            if layer_names in finetune_state_dict:
                finetune_state_dict[layer_names]=torch.tensor(init_state_dict[layer_names])
            else:
                pos=layer_names.find('.')
                for i in range(NUM_CONV):
                    temp_str=layer_names[:pos+1]+str(i)+'.'+layer_names[pos+1:]
                    finetune_state_dict[temp_str]=torch.tensor(init_state_dict[layer_names])
        model_finetune.load_state_dict(finetune_state_dict)
        print("\tReading ->",init_model,"<- success!\n")
    except:
        print("\t!!!we init GCNFF randomly!!!\n")
    model_finetune.to(use_device)
    try:
        optimizer_finetune = torch.optim.Adam([{'params':model_finetune.parameters()},{'params':awl.parameters(),'weight_decay':0}], lr=LEARNING_RATE_INIT)
    except:
        optimizer_finetune = torch.optim.Adam(model_finetune.parameters(), lr=LEARNING_RATE_INIT)
    scheduler_finetune = torch.optim.lr_scheduler.StepLR(optimizer_finetune, step_size=LRStep, gamma=LRGamma)
    
    min_valid_error = np.inf
    train_errors_finetune = list()
    valid_errors_finetune = list()

    training_graph = pickle.load(open(file_path+'/'+'training_graphs.pickle', 'rb'))
    validation_graph = pickle.load(open(file_path+'/'+'validation_graphs.pickle', 'rb'))

    train_size = len(training_graph)
    valid_size = len(validation_graph)
    print("\tthe final model has training data:\t",train_size)
    print("\tthe final model has validation data:\t",valid_size)
    print("\n\n\t\tstarting to training : \n\n")
    if(is_pin_memory=="True"):
        if(Data_shuffle=="True"):
            train_dataloader = DataLoader(training_graph, batch_size=batch_num, pin_memory=True,shuffle=True)
            valid_dataloader = DataLoader(validation_graph, batch_size=batch_num, pin_memory=True,shuffle=True)
        else:
            train_dataloader = DataLoader(training_graph, batch_size=batch_num, pin_memory=True,shuffle=False)
            valid_dataloader = DataLoader(validation_graph, batch_size=batch_num, pin_memory=True,shuffle=False)
    else:
        if(Data_shuffle=="True"):
            train_dataloader = DataLoader(training_graph, batch_size=batch_num, pin_memory=False,shuffle=True)
            valid_dataloader = DataLoader(validation_graph, batch_size=batch_num, pin_memory=False,shuffle=True)
        else:
            train_dataloader = DataLoader(training_graph, batch_size=batch_num, pin_memory=False,shuffle=False)
            valid_dataloader = DataLoader(validation_graph, batch_size=batch_num, pin_memory=False,shuffle=False)
    cnt = 0
    for i in range(max_epoch):
        time_beg_epoch=datetime.datetime.now()
        #training process
        try:
            awl.train()
        except:
            pass
        model_finetune.train()
        train_error=0
        batch_num=0
        for train_graph in train_dataloader:
            batch_num+=1
            train_graph=train_graph.to(use_device)
            optimizer_finetune.zero_grad()
            dist1=torch.index_select(train_graph.pos,0,train_graph.edge_index1[1])-torch.index_select(train_graph.pos,0,train_graph.edge_index1[0])
            dist1=torch.add(dist1,train_graph.edge_attr1)
            try:
                dist2=torch.index_select(train_graph.pos,0,train_graph.edge_index2[1])-torch.index_select(train_graph.pos,0,train_graph.edge_index2[0])
                dist2=torch.add(dist2,train_graph.edge_attr2)
            except:
                dist2=torch.tensor([[],[],[]]).t().to(use_device)
            try:
                pred_energy=model_finetune(train_graph,dist1,dist2,use_device)*100
                pred_force=-torch.autograd.grad(pred_energy,train_graph.pos,
                                                retain_graph=True,create_graph=True,
                                                grad_outputs=torch.ones_like(pred_energy)*100)[0]
                try:
                    true_virial=train_graph.virial
                    pred_fij=-torch.autograd.grad(pred_energy,dist1,
                                                retain_graph=True,create_graph=True,
                                                grad_outputs=torch.ones_like(pred_energy)*100)[0]
                    pred_fjk=-torch.autograd.grad(pred_energy,dist2,
                                                retain_graph=True,create_graph=True,
                                                grad_outputs=torch.ones_like(pred_energy)*100)[0]
                    dist=torch.cat([dist1,dist2]).t()
                    fijk=torch.cat([pred_fij,pred_fjk]).t()
                    pred_v11=torch.sum(dist[0]*fijk[0]).unsqueeze(0).unsqueeze(0)
                    pred_v22=torch.sum(dist[1]*fijk[1]).unsqueeze(0).unsqueeze(0)
                    pred_v33=torch.sum(dist[2]*fijk[2]).unsqueeze(0).unsqueeze(0)
                    pred_v12=0.5*(torch.sum(dist[0]*fijk[1])+torch.sum(dist[1]*fijk[0])).unsqueeze(0).unsqueeze(0)
                    pred_v13=0.5*(torch.sum(dist[0]*fijk[2])+torch.sum(dist[2]*fijk[0])).unsqueeze(0).unsqueeze(0)
                    pred_v23=0.5*(torch.sum(dist[1]*fijk[2])+torch.sum(dist[2]*fijk[1])).unsqueeze(0).unsqueeze(0)
                    pred_v=torch.cat([pred_v11,pred_v22,pred_v33,pred_v12,pred_v13,pred_v23],1)
                    loss_virial=torch.nn.functional.mse_loss(pred_v,true_virial,reduction='none')
                    loss_virial=torch.sum(loss_virial,dim=(0,1),keepdim=True)/6
                    loss_force=torch.norm(train_graph.force-pred_force,p=2,dim=1,keepdim=True)
                    loss_force=torch.mul(loss_force,loss_force)
                    loss_energy=torch.nn.functional.mse_loss(pred_energy,train_graph.y,reduction='none')
                    loss_force_avg=global_mean_pool(loss_force,train_graph.batch)
                    try:
                        total_loss=awl(loss_energy,loss_force_avg,loss_virial)
                    except:
                        total_loss=RHO1*loss_energy+(1-RHO1)*loss_force_avg+RHO2*loss_virial
                    total_loss.backward(torch.ones_like(total_loss))
                    optimizer_finetune.step()
                    train_error+=torch.sum(total_loss).cpu().detach().numpy()
                    if batch_num%batch_step==0:
                        print('   batch ',batch_num,' training error = %.2f'%(total_loss.item()),
                                                    ' loss of energy = %.2f'%(loss_energy.item()),
                                                    ' loss of force = %.2f'%(loss_force_avg.item()),
                                                    'loss of virial = %.2f'%(loss_virial.item()))
                except:
                    loss_force=torch.norm(train_graph.force-pred_force,p=2,dim=1,keepdim=True)
                    loss_force=torch.mul(loss_force,loss_force)
                    loss_energy=torch.nn.functional.mse_loss(pred_energy,train_graph.y,reduction='none')
                    loss_force_avg=global_mean_pool(loss_force,train_graph.batch)
                    try:
                        total_loss=awl(loss_energy,loss_force_avg)
                    except:
                        total_loss=RHO1*loss_energy+(1-RHO1)*loss_force_avg
                    total_loss.backward(torch.ones_like(total_loss))
                    optimizer_finetune.step()
                    train_error+=torch.sum(total_loss).cpu().detach().numpy()
                    if batch_num%batch_step==0:
                        print('   batch ',batch_num,' training error = ',total_loss.item(),
                                                    ' loss of energy = ',loss_energy.item(),
                                                    ' loss of force = ',loss_force_avg.item())
            except:
                torch.cuda.empty_cache()
                print("\tThis training data has  ",train_graph.x.size(0),"\tatoms! May out of memory!")
        train_errors_finetune.append(train_error/train_size)
    
        #validation process
        try:
            awl.eval()
        except:
            pass
        model_finetune.eval()
        optimizer_finetune.zero_grad()
        torch.cuda.empty_cache()
        valid_error=0
        for valid_graph in valid_dataloader:
            valid_graph=valid_graph.to(use_device)
            dist1=torch.index_select(valid_graph.pos,0,valid_graph.edge_index1[1])-torch.index_select(valid_graph.pos,0,valid_graph.edge_index1[0])
            dist1=torch.add(dist1,valid_graph.edge_attr1)
            try:
                dist2=torch.index_select(valid_graph.pos,0,valid_graph.edge_index2[1])-torch.index_select(valid_graph.pos,0,valid_graph.edge_index2[0])
                dist2=torch.add(dist2,valid_graph.edge_attr2)
            except:
                dist2=torch.tensor([[],[],[]]).t().to(use_device)
            try:
                pred_energy=model_finetune(valid_graph,dist1,dist2,use_device)*100
                pred_force=-torch.autograd.grad(pred_energy,valid_graph.pos,
                                                retain_graph=True,create_graph=True,
                                                grad_outputs=torch.ones_like(pred_energy)*100)[0]
                try:
                    true_virial=valid_graph.virial
                    pred_fij=-torch.autograd.grad(pred_energy,dist1,
                                                retain_graph=True,create_graph=True,
                                                grad_outputs=torch.ones_like(pred_energy)*100)[0]
                    pred_fjk=-torch.autograd.grad(pred_energy,dist2,
                                                retain_graph=True,create_graph=True,
                                                grad_outputs=torch.ones_like(pred_energy)*100)[0]
                    dist=torch.cat([dist1,dist2]).t()
                    fijk=torch.cat([pred_fij,pred_fjk]).t()
                    pred_v11=torch.sum(dist[0]*fijk[0]).unsqueeze(0).unsqueeze(0)
                    pred_v22=torch.sum(dist[1]*fijk[1]).unsqueeze(0).unsqueeze(0)
                    pred_v33=torch.sum(dist[2]*fijk[2]).unsqueeze(0).unsqueeze(0)
                    pred_v12=0.5*(torch.sum(dist[0]*fijk[1])+torch.sum(dist[1]*fijk[0])).unsqueeze(0).unsqueeze(0)
                    pred_v13=0.5*(torch.sum(dist[0]*fijk[2])+torch.sum(dist[2]*fijk[0])).unsqueeze(0).unsqueeze(0)
                    pred_v23=0.5*(torch.sum(dist[1]*fijk[2])+torch.sum(dist[2]*fijk[1])).unsqueeze(0).unsqueeze(0)
                    pred_v=torch.cat([pred_v11,pred_v22,pred_v33,pred_v12,pred_v13,pred_v23],1)
                    loss_virial=torch.nn.functional.mse_loss(pred_v,true_virial,reduction='none')
                    loss_virial=torch.sum(loss_virial,dim=(0,1),keepdim=True)/6
                    loss_force=torch.norm(valid_graph.force-pred_force,p=2,dim=1,keepdim=True)
                    loss_force=torch.mul(loss_force,loss_force)
                    loss_energy=torch.nn.functional.mse_loss(pred_energy,valid_graph.y,reduction='none')
                    loss_force_avg=global_mean_pool(loss_force,valid_graph.batch)
                    try:
                        total_loss=awl(loss_energy,loss_force_avg,loss_virial)
                    except:
                        total_loss=RHO1*loss_energy+(1-RHO1)*loss_force_avg+RHO2*loss_virial
                    valid_error+=torch.sum(total_loss).cpu().detach().numpy()
                except:
                    loss_energy=torch.nn.functional.mse_loss(pred_energy,valid_graph.y,reduction='none')
                    loss_force=torch.norm(valid_graph.force-pred_force,p=2,dim=1,keepdim=True)
                    loss_force=torch.mul(loss_force,loss_force)
                    loss_force_avg=global_mean_pool(loss_force,valid_graph.batch)
                    try:
                        total_loss=awl(loss_energy,loss_force_avg)
                    except:
                        total_loss=RHO1*loss_energy+(1-RHO1)*loss_force_avg
                    valid_error+=torch.sum(total_loss).cpu().detach().numpy()
            except:
                torch.cuda.empty_cache()
                print("\tThis validataion data has  ",valid_graph.x.size(0),"\tatoms! May out of memory!")
        valid_errors_finetune.append(valid_error/valid_size)
    
        #print information & judgement for early stopping
        scheduler_finetune.step()
        time_end_epoch=datetime.datetime.now()
        print('Epoch ',i,' training error = ',train_error/train_size,
            ' validation error = ',valid_error/valid_size,
            ' training and validation time = ',time_end_epoch-time_beg_epoch,
            ' learning rate = ',optimizer_finetune.param_groups[0]["lr"])

        with open('Epoches-Errors-finetune.txt', 'w') as f:
            f.write('Epoches\t\tTraining_Error\t\tValidation_Error\n')
            for index in range(len(train_errors_finetune)):
                f.write("%d\t\t%lf\t\t%lf\n" % (index, train_errors_finetune[index], valid_errors_finetune[index]))

        if valid_error<min_valid_error: #judgement for early stopping
            cnt=0
            try:
                torch.save(awl,'AutomaticWeightedLoss_finetune.pkl')
            except:
                pass
            torch.save(model_finetune,'best_model_finetune.pkl')
            min_valid_error=valid_error
        else:
            cnt+=1
            if cnt>=CNT:
                print('Early stopping')
                try:
                    del(awl)
                except:
                    pass
                del(model_finetune)
                #with open('training_errors_finetune.pickle','wb') as f:
                #    pickle.dump(train_errors_finetune,f)
                #with open('valid_errors_finetune.pickle','wb') as f:
                #    pickle.dump(valid_errors_finetune,f)
                break
    print("\n\t------------------------------------------------\n")
    

def direct_train(config_file):
    config = load_file(config_file)
    
    print("\n\t----------------Another-DataLoader-Method---------------------\n")
    print("\tTraining an 'final model' with unfixed parameters in Conv layers without divide_set.\n")
    try:
        RHO1 = config['training']['RHO1']
    except:
        RHO1 = 0.01
        print("\tYou did not set the value of ->['training']['RHO1']<-, and it has been set to 0.01 by default!\n")
    try:
        RHO2 = config['training']['RHO2']
    except:
        RHO2 = 0.001
        print("\tYou did not set the value of ->['training']['RHO2']<-, and it has been set to 0.001 by default!\n")
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
        LEARNING_RATE_INIT = config['training']['LEARNING_RATE_INIT']
    except:
        LEARNING_RATE_INIT = 0.0002
        print("\tYou did not set the value of ->['training']['LEARNING_RATE_INIT']<-, and it has been set to 0.0002 by default!\n")
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
    try:
        file_path = config['training']['file_path']
    except:
        file_path = "./model_graphdata"
        print("\tYou did not set the value of ->['training']['file_path']<-, and it has been set to './model_graphdata' by default!\n")
    try:
        use_device = config['training']['use_device']
    except:
        use_device= 'cuda:0' if torch.cuda.is_available() else 'cpu' 
        print("\tYou did not set the value of ->['training']['use_device']<-, and it has been set to ",use_device," by default!\n")
    try:
        is_pin_memory = config['training']['pin_memory']
    except:
        is_pin_memory="False"
        print("\tYou did not set the value of ->['training']['pin_memory']<-, and it has been set to 'False' by default!\n")
    try:
        batch_num = config['training']['batch_num']
        if(batch_num!=1):
            print("\tGCNFF only supports the setting of ['training']['batch_num'] to 1 currently\n")
        batch_num=1
    except:
        batch_num=1
        print("\t->['training']['batch_num']<- has been set to 1 by default!\n")
    try:
        LRStep = config['training']['LRStep']
    except:
        LRStep =5
        print("\tYou did not set the value of ->['training']['LRStep']<-, and it has been set to 5 by default!\n")
    try:
        LRGamma = config['training']['LRGamma']
    except:
        LRGamma =0.8
        print("\tYou did not set the value of ->['training']['LRGamma']<-, and it has been set to 0.8 by default!\n")
    try:
        CNT = config['training']['CNT']
    except:
        CNT = 10
        print("\tYou did not set the value of ->['training']['CNT']<-, and it has been set to 10 by default!\n")
    try:
        batch_step = config['training']['batch_step']
    except:
        batch_step = 1000
        print("\tYou did not set the value of ->['training']['batch_step']<-, and it has been set to 1000 by default!\n")
    try:
        max_epoch = config['training']['max_epoch']
    except:
        max_epoch = 300
        print("\tYou did not set the value of ->['training']['max_epoch']<-, and it has been set to 300 by default!\n")
    try:
        Data_shuffle = config['training']['Data_shuffle']
    except:
        Data_shuffle ="False"
        print("\tYou did not set the value of ->['training']['Data_shuffle']<-, and it has been set to 'False' by default!\n")
    try:
        Flag_AutomaticWeightedLoss = config['training']['Flag_AutoLoss']
    except:
        Flag_AutomaticWeightedLoss ="False"
        print("\tYou did not set the value of ->['training']['Flag_AutoLoss']<-, and it has been set to 'False' by default!\n")
        
    try:
        init_model=config['direct_train']['begin_model']
    except:
        print("\tYou did not set the value of ->['direct_train']['begin_model']<- !\n")
    try:
        randomSeed=config['direct_train']['RandomSeed']
    except:
        randomSeed=0
        print("\tYou did not set the value of ->['direct_train']['RandomSeed']<-, and it has been set to 0 by default!\n")
    try:
        file_path = config['direct_train']['graphfile_path']
    except:
        file_path = "./graphdata"
        print("\tYou did not set the value of ->['direct_train']['graphfile_path']<-, and it has been set to './graphdata/' by default!\n")
    try:
        traindate_for_finalmodel=config['direct_train']['traindata']
    except:
        traindate_for_finalmodel=0.7
        print("\tYou did not set the value of ->['direct_train']['traindata']<-, and it has been set to 0.7 by default!\n")
    if (Flag_AutomaticWeightedLoss=="True"):
        awl = AutomaticWeightedLoss(3)
        try:
            awl_init=torch.load("./AutomaticWeightedLoss.pkl",map_location='cpu')
            init_awl_state_dict=awl_init.state_dict()
            awl.load_state_dict(init_awl_state_dict)
            print("\tReading ->AutomaticWeightedLoss.pkl<- success!\n")
        except:
            print("\tCannot reading ->AutomaticWeightedLoss.pkl<- , so we init weight of AutomaticWeightedLoss as ones\n")
        awl.to(use_device)
        print("\tUsing the AutomaticWeightedLoss method! ->https://github.com/Mikoto10032/AutomaticWeightedLoss<-\n")
        
    model_finetune=GCNFF(cutoff1=CUTOFF_DISTANCE1,cutoff2=CUTOFF_DISTANCE2,gamma=GAMMA,rbfkernel_number=RBF_KERNEL_NUM,
                     hidden_layer_dimensions=HID_DIM,num_conv=NUM_CONV,atom_types=ATOM_TYPES,exponent=EXPONENT)
    try:
        model_init=torch.load(init_model,map_location='cpu')
        init_state_dict=model_init.state_dict()
        finetune_state_dict=model_finetune.state_dict()
        for layer_names in init_state_dict:
            if layer_names in finetune_state_dict:
                finetune_state_dict[layer_names]=torch.tensor(init_state_dict[layer_names])
            else:
                pos=layer_names.find('.')
                for i in range(NUM_CONV):
                    temp_str=layer_names[:pos+1]+str(i)+'.'+layer_names[pos+1:]
                    finetune_state_dict[temp_str]=torch.tensor(init_state_dict[layer_names])
        model_finetune.load_state_dict(finetune_state_dict)
        print("\tReading ->",init_model,"<- success!\n")
    except:
        print("\t!!!we init GCNFF randomly!!!\n")
    model_finetune.to(use_device)
    try:
        optimizer_finetune = torch.optim.Adam([{'params':model_finetune.parameters()},{'params':awl.parameters(),'weight_decay':0}], lr=LEARNING_RATE_INIT)
    except:
        optimizer_finetune = torch.optim.Adam(model_finetune.parameters(), lr=LEARNING_RATE_INIT)
    scheduler_finetune = torch.optim.lr_scheduler.StepLR(optimizer_finetune, step_size=LRStep, gamma=LRGamma)
    
    min_valid_error = np.inf
    train_errors_finetune = list()
    valid_errors_finetune = list()
    cnt = 0
    file_names=os.listdir(file_path)
    for i in range(max_epoch):
        time_beg_epoch=datetime.datetime.now()
        #training process
        try:
            awl.train()
        except:
            pass
        model_finetune.train()
        train_error=0
        num_batch=0
        for file in file_names:
            graph_list=pickle.load(open(file_path+'/'+file,'rb'))
            random.seed(randomSeed)
            random.shuffle(graph_list)
            N_training = int(traindate_for_finalmodel * len(graph_list))
            train_size = len(graph_list[:N_training])
            valid_size = len(graph_list[N_training:])
            print("\tRead the ->",file,"<- success! Total configuration:",len(graph_list))
            if(is_pin_memory=="True"):
                train_dataloader = DataLoader(graph_list[:N_training], batch_size=batch_num, pin_memory=True)
                valid_dataloader = DataLoader(graph_list[N_training:], batch_size=batch_num, pin_memory=True)
            else:
                train_dataloader = DataLoader(graph_list[:N_training], batch_size=batch_num, pin_memory=False)
                valid_dataloader = DataLoader(graph_list[N_training:], batch_size=batch_num, pin_memory=False)

            for train_graph in train_dataloader:
                num_batch+=1
                train_graph=train_graph.to(use_device)
                optimizer_finetune.zero_grad()
                dist1=torch.index_select(train_graph.pos,0,train_graph.edge_index1[1])-torch.index_select(train_graph.pos,0,train_graph.edge_index1[0])
                dist1=torch.add(dist1,train_graph.edge_attr1)
                try:
                    dist2=torch.index_select(train_graph.pos,0,train_graph.edge_index2[1])-torch.index_select(train_graph.pos,0,train_graph.edge_index2[0])
                    dist2=torch.add(dist2,train_graph.edge_attr2)
                except:
                    dist2=torch.tensor([[],[],[]]).t().to(use_device)
                try:
                    pred_energy=model_finetune(train_graph,dist1,dist2,use_device)*100
                    pred_force=-torch.autograd.grad(pred_energy,train_graph.pos,
                                                    retain_graph=True,create_graph=True,
                                                    grad_outputs=torch.ones_like(pred_energy)*100)[0]
                    try:
                        true_virial=train_graph.virial
                        pred_fij=-torch.autograd.grad(pred_energy,dist1,
                                                    retain_graph=True,create_graph=True,
                                                    grad_outputs=torch.ones_like(pred_energy)*100)[0]
                        pred_fjk=-torch.autograd.grad(pred_energy,dist2,
                                                    retain_graph=True,create_graph=True,
                                                    grad_outputs=torch.ones_like(pred_energy)*100)[0]
                        dist=torch.cat([dist1,dist2]).t()
                        fijk=torch.cat([pred_fij,pred_fjk]).t()
                        pred_v11=torch.sum(dist[0]*fijk[0]).unsqueeze(0).unsqueeze(0)
                        pred_v22=torch.sum(dist[1]*fijk[1]).unsqueeze(0).unsqueeze(0)
                        pred_v33=torch.sum(dist[2]*fijk[2]).unsqueeze(0).unsqueeze(0)
                        pred_v12=0.5*(torch.sum(dist[0]*fijk[1])+torch.sum(dist[1]*fijk[0])).unsqueeze(0).unsqueeze(0)
                        pred_v13=0.5*(torch.sum(dist[0]*fijk[2])+torch.sum(dist[2]*fijk[0])).unsqueeze(0).unsqueeze(0)
                        pred_v23=0.5*(torch.sum(dist[1]*fijk[2])+torch.sum(dist[2]*fijk[1])).unsqueeze(0).unsqueeze(0)
                        pred_v=torch.cat([pred_v11,pred_v22,pred_v33,pred_v12,pred_v13,pred_v23],1)
                        loss_virial=torch.nn.functional.mse_loss(pred_v,true_virial,reduction='none')
                        loss_virial=torch.sum(loss_virial,dim=(0,1),keepdim=True)/6
                        loss_force=torch.norm(train_graph.force-pred_force,p=2,dim=1,keepdim=True)
                        loss_force=torch.mul(loss_force,loss_force)
                        loss_energy=torch.nn.functional.mse_loss(pred_energy,train_graph.y,reduction='none')
                        loss_force_avg=global_mean_pool(loss_force,train_graph.batch)
                        try:
                            total_loss=awl(loss_energy,loss_force_avg,loss_virial)
                        except:
                            total_loss=RHO1*loss_energy+(1-RHO1)*loss_force_avg+RHO2*loss_virial
                        total_loss.backward(torch.ones_like(total_loss))
                        optimizer_finetune.step()
                        train_error+=torch.sum(total_loss).cpu().detach().numpy()
                        if num_batch%batch_step==0:
                            print('   batch ',num_batch,' training error = %.2f'%(total_loss.item()),
                                                        ' loss of energy = %.2f'%(loss_energy.item()),
                                                        ' loss of force = %.2f'%(loss_force_avg.item()),
                                                        'loss of virial = %.2f'%(loss_virial.item()))
                    except:
                        loss_force=torch.norm(train_graph.force-pred_force,p=2,dim=1,keepdim=True)
                        loss_force=torch.mul(loss_force,loss_force)
                        loss_energy=torch.nn.functional.mse_loss(pred_energy,train_graph.y,reduction='none')
                        loss_force_avg=global_mean_pool(loss_force,train_graph.batch)
                        try:
                            total_loss=awl(loss_energy,loss_force_avg)
                        except:
                            total_loss=RHO1*loss_energy+(1-RHO1)*loss_force_avg
                        total_loss.backward(torch.ones_like(total_loss))
                        optimizer_finetune.step()
                        train_error+=torch.sum(total_loss).cpu().detach().numpy()
                        if num_batch%batch_step==0:
                            print('   batch ',num_batch,' training error = ',total_loss.item(),
                                                        ' loss of energy = ',loss_energy.item(),
                                                        ' loss of force = ',loss_force_avg.item())
                except:
                    torch.cuda.empty_cache()
                    print("\tThis training data has  ",train_graph.x.size(0),"\tatoms! May out of memory!")
            train_errors_finetune.append(train_error/train_size)
    
            #validation process
            try:
                awl.eval()
            except:
                pass
            model_finetune.eval()
            optimizer_finetune.zero_grad()
            torch.cuda.empty_cache()
            valid_error=0
            for valid_graph in valid_dataloader:
                valid_graph=valid_graph.to(use_device)
                dist1=torch.index_select(valid_graph.pos,0,valid_graph.edge_index1[1])-torch.index_select(valid_graph.pos,0,valid_graph.edge_index1[0])
                dist1=torch.add(dist1,valid_graph.edge_attr1)
                try:
                    dist2=torch.index_select(valid_graph.pos,0,valid_graph.edge_index2[1])-torch.index_select(valid_graph.pos,0,valid_graph.edge_index2[0])
                    dist2=torch.add(dist2,valid_graph.edge_attr2)
                except:
                    dist2=torch.tensor([[],[],[]]).t().to(use_device)
                try:
                    pred_energy=model_finetune(valid_graph,dist1,dist2,use_device)*100
                    pred_force=-torch.autograd.grad(pred_energy,valid_graph.pos,
                                                    retain_graph=True,create_graph=True,
                                                    grad_outputs=torch.ones_like(pred_energy)*100)[0]
                    try:
                        true_virial=valid_graph.virial
                        pred_fij=-torch.autograd.grad(pred_energy,dist1,
                                                    retain_graph=True,create_graph=True,
                                                    grad_outputs=torch.ones_like(pred_energy)*100)[0]
                        pred_fjk=-torch.autograd.grad(pred_energy,dist2,
                                                    retain_graph=True,create_graph=True,
                                                    grad_outputs=torch.ones_like(pred_energy)*100)[0]
                        dist=torch.cat([dist1,dist2]).t()
                        fijk=torch.cat([pred_fij,pred_fjk]).t()
                        pred_v11=torch.sum(dist[0]*fijk[0]).unsqueeze(0).unsqueeze(0)
                        pred_v22=torch.sum(dist[1]*fijk[1]).unsqueeze(0).unsqueeze(0)
                        pred_v33=torch.sum(dist[2]*fijk[2]).unsqueeze(0).unsqueeze(0)
                        pred_v12=0.5*(torch.sum(dist[0]*fijk[1])+torch.sum(dist[1]*fijk[0])).unsqueeze(0).unsqueeze(0)
                        pred_v13=0.5*(torch.sum(dist[0]*fijk[2])+torch.sum(dist[2]*fijk[0])).unsqueeze(0).unsqueeze(0)
                        pred_v23=0.5*(torch.sum(dist[1]*fijk[2])+torch.sum(dist[2]*fijk[1])).unsqueeze(0).unsqueeze(0)
                        pred_v=torch.cat([pred_v11,pred_v22,pred_v33,pred_v12,pred_v13,pred_v23],1)
                        loss_virial=torch.nn.functional.mse_loss(pred_v,true_virial,reduction='none')
                        loss_virial=torch.sum(loss_virial,dim=(0,1),keepdim=True)/6
                        loss_force=torch.norm(valid_graph.force-pred_force,p=2,dim=1,keepdim=True)
                        loss_force=torch.mul(loss_force,loss_force)
                        loss_energy=torch.nn.functional.mse_loss(pred_energy,valid_graph.y,reduction='none')
                        loss_force_avg=global_mean_pool(loss_force,valid_graph.batch)
                        try:
                            total_loss=awl(loss_energy,loss_force_avg,loss_virial)
                        except:
                            total_loss=RHO1*loss_energy+(1-RHO1)*loss_force_avg+RHO2*loss_virial
                        valid_error+=torch.sum(total_loss).cpu().detach().numpy()
                    except:
                        loss_energy=torch.nn.functional.mse_loss(pred_energy,valid_graph.y,reduction='none')
                        loss_force=torch.norm(valid_graph.force-pred_force,p=2,dim=1,keepdim=True)
                        loss_force=torch.mul(loss_force,loss_force)
                        loss_force_avg=global_mean_pool(loss_force,valid_graph.batch)
                        try:
                            total_loss=awl(loss_energy,loss_force_avg)
                        except:
                            total_loss=RHO1*loss_energy+(1-RHO1)*loss_force_avg
                        valid_error+=torch.sum(total_loss).cpu().detach().numpy()
                except:
                    torch.cuda.empty_cache()
                    print("\tThis validataion data has  ",valid_graph.x.size(0),"\tatoms! May out of memory!")
            valid_errors_finetune.append(valid_error/valid_size)
    
        #print information & judgement for early stopping
        scheduler_finetune.step()
        time_end_epoch=datetime.datetime.now()
        print('Epoch ',i,' training error = ',train_error/train_size,
            ' validation error = ',valid_error/valid_size,
            ' training and validation time = ',time_end_epoch-time_beg_epoch,
            ' learning rate = ',optimizer_finetune.param_groups[0]["lr"])

        with open('Epoches-Errors-finetune.txt', 'w') as f:
            f.write('Epoches\t\tTraining_Error\t\tValidation_Error\n')
            for index in range(len(train_errors_finetune)):
                f.write("%d\t\t%lf\t\t%lf\n" % (index, train_errors_finetune[index], valid_errors_finetune[index]))

        if valid_error<min_valid_error: #judgement for early stopping
            cnt=0
            try:
                torch.save(awl,'AutomaticWeightedLoss_finetune.pkl')
            except:
                pass
            torch.save(model_finetune,'best_model_finetune.pkl')
            min_valid_error=valid_error
        else:
            cnt+=1
            if cnt>=CNT:
                print('Early stopping')
                try:
                    del(awl)
                except:
                    pass
                del(model_finetune)
                break
    print("\n\t------------------------------------------------\n")
    
def model_test(config_file):
    config = load_file(config_file)
    print("\n\t------------------Step--Five---------------------\n")
    try:
        use_device = config['model_test']['use_device']
    except:
        use_device= 'cuda:0' if torch.cuda.is_available() else 'cpu' 
        print("\tYou did not set the value of ->['model_test']['use_device']<-, and it has been set to ",use_device," by default!\n")
    try:
        is_pin_memory = config['training']['pin_memory']
    except:
        is_pin_memory="False"
        print("\tYou did not set the value of ->['training']['pin_memory']<-, and it has been set to 'False' by default!\n")
    try:
        batch_num = config['training']['batch_num']
        if(batch_num!=1):
            print("\tGCNFF only supports the setting of ['training']['batch_num'] to 1 currently\n")
        batch_num=1
    except:
        batch_num=1
        print("\t->['training']['batch_num']<- has been set to 1 by default!\n")
    try:
        testdata = config['model_test']['testdata']
    except:
        print("\tThe test dataset ->['model_test']['testdata']<- must be specified by yourself!\n")
        exit(1)
    try:
        testmodel= config['model_test']['testmodel']
    except:
        print("\tThe test model ->['model_test']['testmodel']<- must be specified by yourself!\n")
        exit(1)
    print("\tUsing the -> ",testdata," <- to test the -> ",testmodel," <- .\n")
    test_graph = pickle.load(open(testdata, 'rb'))
    test_size = len(test_graph)
    print("\tthe test data has :\t",test_size)
    test_dataloader = DataLoader(test_graph, batch_size=batch_num, pin_memory=is_pin_memory)
    final_model = torch.load(testmodel,map_location='cpu')
    final_model = final_model.to(use_device)
    final_model.eval()
    pred_energies=list()
    pred_forces=list()
    pred_virials=list()
    true_energies=list()
    true_forces=list()
    true_virials=list()
    no_virial_n=0
    for test_graph in test_dataloader:
        test_graph=test_graph.to(use_device)
        dist1=torch.index_select(test_graph.pos,0,test_graph.edge_index1[1])-torch.index_select(test_graph.pos,0,test_graph.edge_index1[0])
        dist1=torch.add(dist1,test_graph.edge_attr1)
        try:
            dist2=torch.index_select(test_graph.pos,0,test_graph.edge_index2[1])-torch.index_select(test_graph.pos,0,test_graph.edge_index2[0])
            dist2=torch.add(dist2,test_graph.edge_attr2)
        except:
            dist2=torch.tensor([[],[],[]]).t().to(use_device) 
        try:
            pred_energy=final_model(test_graph,dist1,dist2,use_device)*100
            pred_force=-torch.autograd.grad(pred_energy,test_graph.pos,
                                            retain_graph=True,create_graph=True,
                                            grad_outputs=torch.ones_like(pred_energy)*100)[0]
        
            pred_energies.append(pred_energy.cpu().detach().numpy()/test_graph.x.size(0))
            pred_forces.append(pred_force.cpu().detach().numpy())
            true_energies.append(test_graph.y.cpu().detach().numpy()/test_graph.x.size(0))
            true_forces.append(test_graph.force.cpu().detach().numpy())
            try:
                true_virial=test_graph.virial
                pred_fij=-torch.autograd.grad(pred_energy,dist1,
                                            retain_graph=True,create_graph=True,
                                            grad_outputs=torch.ones_like(pred_energy)*100)[0]
                pred_fjk=-torch.autograd.grad(pred_energy,dist2,
                                            retain_graph=True,create_graph=True,
                                            grad_outputs=torch.ones_like(pred_energy)*100)[0]
                dist=torch.cat([dist1,dist2]).t()
                fijk=torch.cat([pred_fij,pred_fjk]).t()
                pred_v11=torch.sum(dist[0]*fijk[0]).unsqueeze(0).unsqueeze(0)
                pred_v22=torch.sum(dist[1]*fijk[1]).unsqueeze(0).unsqueeze(0)
                pred_v33=torch.sum(dist[2]*fijk[2]).unsqueeze(0).unsqueeze(0)
                pred_v12=0.5*(torch.sum(dist[0]*fijk[1])+torch.sum(dist[1]*fijk[0])).unsqueeze(0).unsqueeze(0)
                pred_v13=0.5*(torch.sum(dist[0]*fijk[2])+torch.sum(dist[2]*fijk[0])).unsqueeze(0).unsqueeze(0)
                pred_v23=0.5*(torch.sum(dist[1]*fijk[2])+torch.sum(dist[2]*fijk[1])).unsqueeze(0).unsqueeze(0)
                pred_v=torch.cat([pred_v11,pred_v22,pred_v33,pred_v12,pred_v13,pred_v23],1)
                true_virials.append(true_virial.cpu().detach().numpy())
                pred_virials.append(pred_v.cpu().detach().numpy())
            except:
                no_virial_n+=1
        except:
            torch.cuda.empty_cache()
            print("\tThis test data has  ",test_graph.x.size(0),"\tatoms! May out of memory!")
    print("\tThere are ",no_virial_n,"datas have no virial information!\n")
    pred_energies=np.concatenate(pred_energies,axis=0)
    pred_forces=np.concatenate(pred_forces,axis=0)
    
    true_energies=np.concatenate(true_energies,axis=0)
    true_forces=np.concatenate(true_forces,axis=0)
    print('\tMAE of Energy in test data: ',1000*np.mean(np.fabs(pred_energies-true_energies)),' meV/atom ')
    print('\tRMSE of Energy in test data: ',1000*np.sqrt(np.mean(np.square(pred_energies-true_energies))),' meV/atom\n')
    print('\tMAE of Force in test data: ',(1/3)*(np.mean(np.linalg.norm(pred_forces-true_forces,ord=1,axis=1))),' eV/Å ')
    print('\tRMSE of Force in test data: ',np.sqrt((1/3)*(np.mean(np.linalg.norm(pred_forces-true_forces,ord=2,axis=1)**2))),'eV/Å\n')
    try:
        pred_virials=np.concatenate(pred_virials,axis=0)
        true_virials=np.concatenate(true_virials,axis=0)
        print('\tMAE of Virial in test data: ',(1/6)*np.mean(np.linalg.norm(pred_virials-true_virials,ord=1,axis=1)),' eV ')
        print('\tRMSE of Virial in test data: ',np.sqrt((1/6)*np.mean(np.linalg.norm(pred_virials-true_virials,ord=2,axis=1)**2)),'eV \n')
        virial=np.concatenate((pred_virials.reshape((-1,1)),true_virials.reshape((-1,1))),axis=1)
        np.savetxt("./pred-ture_virial.txt",virial,fmt='%lf',delimiter=' ')
    except:
        print('\tno virial information!')

    energy=np.concatenate((pred_energies,true_energies),axis=1)
    np.savetxt("./pred-ture_energy.txt",energy,fmt='%lf',delimiter=' ')
    force=np.concatenate((pred_forces.reshape((-1,1)),true_forces.reshape((-1,1))),axis=1)
    np.savetxt("./pred-ture_force.txt",force,fmt='%lf',delimiter=' ')
    print("\n\t------------------------------------------------\n")

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

def main():
    if len(sys.argv) < 2:
        showhelp()
        exit(-1)
    action = sys.argv[1]
    if action == '-help' or action == '-h':
        showhelp()
        exit(0)
    if action not in ['get_graph', 'divide_set', 'init_train', 'final_train', 'model_test', 'get_potential','showfig','direct_train','-help']:
        showhelp()
        exit(-1)
    sys.stdout=Logger('gcnff.log')
    config_file = sys.argv[2]
    if action == 'get_graph':
        get_graph(config_file)

    if action == 'divide_set':
        divide_set(config_file)

    if action == 'init_train':
        init_train(config_file)

    if action == 'final_train':
        final_train(config_file)

    if action == 'direct_train':
        direct_train(config_file)
        
    if action == 'model_test':
        model_test(config_file)

    if action == 'get_potential':
        get_potential(config_file)
        
    if action == 'showfig':
        showfig(config_file)
        
if __name__ == '__main__':
    main()
