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
from gcnff.model import Schnet_init, global_mean_pool, Schnet
from gcnff.printsave import Logger
from gcnff.showplot import showfig

def load_file(config):
    f = open(config)
    return json.load(f)

def get_graph(config_file):
    config = load_file(config_file)
    CUTOFF_DISTANCE1 = config['get_graph']['CUTOFF_DISTANCE1']
    CUTOFF_DISTANCE2 = config['get_graph']['CUTOFF_DISTANCE2']
    file_path = config['get_graph']['file_path']
    jumpNum = config['get_graph']['jumpNum']
    baseEvalue = config['get_graph']['baseEvalue']
    
    print("\n\t------------------Step--One---------------------\n")
    print("\tCreating a folder 'graphdata' in the current directory \n\t\tto store the graph format of each input file.\n")
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
    
    randomSeed=config['divide_set']['RandomSeed']
    file_path = config['divide_set']['graphfile_path']
    date_for_initmodel=config['divide_set']['initmodel_data']
    traindate_for_initmodel=config['divide_set']['initmodel_traindata']
    traindate_for_finalmodel=config['divide_set']['finalmodel_traindata']
    validate_for_finalmodel=config['divide_set']['finalmodel_validata']
    
    print("\n\t------------------Step--Two---------------------\n")
    print("\tCreating a folder 'model_graphdata' in the current directory \n\t\tto store the divided data set.\n")
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
    print("\tThe total number of graph structures read is:",len(graph_list))
    
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
    
    RHO1 = config['training']['RHO1']
    RHO2 = config['training']['RHO2']
    GAMMA = config['training']['GAMMA']
    RBF_KERNEL_NUM = config['training']['RBF_KERNEL_NUM']
    HID_DIM = config['training']['HID_DIM']
    NUM_CONV = config['training']['NUM_CONV']
    LEARNING_RATE_INIT = config['training']['LEARNING_RATE_INIT']
    ATOM_TYPES = config['training']['ATOM_TYPES']
    CUTOFF_DISTANCE1 = config['get_graph']['CUTOFF_DISTANCE1']
    CUTOFF_DISTANCE2 = config['get_graph']['CUTOFF_DISTANCE2']
    EXPONENT = config['training']['EXPONENT']
    file_path = config['training']['file_path']
    use_device = config['training']['use_device']
    is_pin_memory = config['training']['pin_memory']
    batch_num = config['training']['batch_num']
    LRStep = config['training']['LRStep']
    LRGamma = config['training']['LRGamma']
    CNT = config['training']['CNT']
    batch_step = config['training']['batch_step']
    max_epoch = config['training']['max_epoch']
    
    print("\n\t------------------Step--Three---------------------\n")
    print("\tTraining an 'initial model' with fixed parameters in Conv layers.\n")
    
    init_training_graph = pickle.load(open(file_path+'/'+'init_training_graphs.pickle', 'rb'))
    init_validation_graph = pickle.load(open(file_path+'/'+'init_validation_graphs.pickle', 'rb'))

    init_train_size = len(init_training_graph)
    init_valid_size = len(init_validation_graph)
    print("\tthe init model has training data:\t",init_train_size)
    print("\tthe init model has validation data:\t",init_valid_size)
    print("\n\n\t\tstarting to training : \n\n")
    init_train_dataloader = DataLoader(init_training_graph, batch_size=batch_num, pin_memory=is_pin_memory)
    init_valid_dataloader = DataLoader(init_validation_graph, batch_size=batch_num, pin_memory=is_pin_memory)
    model_init=Schnet_init(cutoff1=CUTOFF_DISTANCE1,cutoff2=CUTOFF_DISTANCE2,gamma=GAMMA,rbfkernel_number=RBF_KERNEL_NUM,
                            hidden_layer_dimensions=HID_DIM,num_conv=NUM_CONV,atom_types=ATOM_TYPES,exponent=EXPONENT)

    model_init.to(use_device)
    optimizer_init = torch.optim.Adam(model_init.parameters(), lr=LEARNING_RATE_INIT)
    scheduler_init = torch.optim.lr_scheduler.StepLR(optimizer_init, step_size=LRStep, gamma=LRGamma)

    min_valid_error=np.inf
    train_errors_init=list()
    valid_errors_init=list()
    cnt=0
    for i in range(max_epoch):
        time_beg_epoch=datetime.datetime.now()
        #training process
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
                    total_loss=RHO1*loss_energy+(1-RHO1)*loss_force_avg+RHO2*loss_virial
                    valid_error+=torch.sum(total_loss).cpu().detach().numpy()
                except:
                    loss_energy=torch.nn.functional.mse_loss(pred_energy,valid_graph.y,reduction='none')
                    loss_force=torch.norm(valid_graph.force-pred_force,p=2,dim=1,keepdim=True)
                    loss_force=torch.mul(loss_force,loss_force)
                    loss_force_avg=global_mean_pool(loss_force,valid_graph.batch)
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
    
        if valid_error<min_valid_error: #judgement for early stopping
            cnt=0
            torch.save(model_init,'best_model_init.pkl')
            min_valid_error=valid_error
        else:
            cnt+=1
            if cnt>=CNT:
                print('Early stopping')
                del(model_init)
                #with open('training_errors_init.pickle','wb') as f:
                #    pickle.dump(train_errors_init,f)
                #with open('valid_errors_init.pickle','wb') as f:
                #    pickle.dump(valid_errors_init,f)
                with open('Epoches-Errors-init.txt', 'w') as f:
                    f.write('Epoches\t\tTraining_Error\t\tValidation_Error\n')
                    for index in range(len(train_errors_init)):
                        f.write("%d\t\t%lf\t\t%lf\n" % (index, train_errors_init[index], valid_errors_init[index]))
                break
    print("\n\t------------------------------------------------\n")


def final_train(config_file):
    config = load_file(config_file)
    
    RHO1 = config['training']['RHO1']
    RHO2 = config['training']['RHO2']
    GAMMA = config['training']['GAMMA']
    RBF_KERNEL_NUM = config['training']['RBF_KERNEL_NUM']
    HID_DIM = config['training']['HID_DIM']
    NUM_CONV = config['training']['NUM_CONV']
    LEARNING_RATE_INIT = config['training']['LEARNING_RATE_INIT']
    ATOM_TYPES = config['training']['ATOM_TYPES']
    CUTOFF_DISTANCE1 = config['get_graph']['CUTOFF_DISTANCE1']
    CUTOFF_DISTANCE2 = config['get_graph']['CUTOFF_DISTANCE2']
    EXPONENT = config['training']['EXPONENT']
    file_path = config['training']['file_path']
    use_device = config['training']['use_device']
    is_pin_memory = config['training']['pin_memory']
    batch_num = config['training']['batch_num']
    LRStep = config['training']['LRStep']
    LRGamma = config['training']['LRGamma']
    CNT = config['training']['CNT']
    batch_step = config['training']['batch_step']
    init_model=config['final_model']['begin_model']
    max_epoch = config['training']['max_epoch']
    
    print("\n\t------------------Step--Four---------------------\n")
    print("\tRetraining an 'final model' with unfixed parameters in Conv layers.\n")

    model_finetune=Schnet(cutoff1=CUTOFF_DISTANCE1,cutoff2=CUTOFF_DISTANCE2,gamma=GAMMA,rbfkernel_number=RBF_KERNEL_NUM,
                     hidden_layer_dimensions=HID_DIM,num_conv=NUM_CONV,atom_types=ATOM_TYPES,exponent=EXPONENT)
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
    
    model_finetune.to(use_device)
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
    train_dataloader = DataLoader(training_graph, batch_size=batch_num, pin_memory=is_pin_memory)
    valid_dataloader = DataLoader(validation_graph, batch_size=batch_num, pin_memory=is_pin_memory)
    
    cnt = 0
    for i in range(max_epoch):
        time_beg_epoch=datetime.datetime.now()
        #training process
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
                    total_loss=RHO1*loss_energy+(1-RHO1)*loss_force_avg+RHO2*loss_virial
                    valid_error+=torch.sum(total_loss).cpu().detach().numpy()
                except:
                    loss_energy=torch.nn.functional.mse_loss(pred_energy,valid_graph.y,reduction='none')
                    loss_force=torch.norm(valid_graph.force-pred_force,p=2,dim=1,keepdim=True)
                    loss_force=torch.mul(loss_force,loss_force)
                    loss_force_avg=global_mean_pool(loss_force,valid_graph.batch)
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
    
        if valid_error<min_valid_error: #judgement for early stopping
            cnt=0
            torch.save(model_finetune,'best_model_finetune.pkl')
            min_valid_error=valid_error
        else:
            cnt+=1
            if cnt>=CNT:
                print('Early stopping')
                del(model_finetune)
                with open('Epoches-Errors-finetune.txt', 'w') as f:
                    f.write('Epoches\t\tTraining_Error\t\tValidation_Error\n')
                    for index in range(len(train_errors_finetune)):
                        f.write("%d\t\t%lf\t\t%lf\n" % (index, train_errors_finetune[index], valid_errors_finetune[index]))
                #with open('training_errors_finetune.pickle','wb') as f:
                #    pickle.dump(train_errors_finetune,f)
                #with open('valid_errors_finetune.pickle','wb') as f:
                #    pickle.dump(valid_errors_finetune,f)
                break
    print("\n\t------------------------------------------------\n")
    
def model_test(config_file):
    config = load_file(config_file)
    testdata = config['model_test']['testdata']
    use_device = config['model_test']['use_device']
    testmodel= config['model_test']['testmodel']
    is_pin_memory = config['training']['pin_memory']
    batch_num = config['training']['batch_num']
    
    print("\n\t------------------Step--Five---------------------\n")
    print("\tUsing the 'specified test set' to test the 'specified model'.\n")
    
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
    print('\tMAE of Energy in test data: ',np.mean(np.fabs(pred_energies-true_energies)))
    print('\tMAE of Force in test data: ',(1/3)*(np.mean(np.linalg.norm(pred_forces-true_forces,ord=1,axis=1))))
    try:
        pred_virials=np.concatenate(pred_virials,axis=0)
        true_virials=np.concatenate(true_virials,axis=0)
        print('\tMAE of Virial in test data: ',(1/6)*np.mean(np.linalg.norm(pred_virials-true_virials,ord=1,axis=1)))
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
    
    GAMMA = config['training']['GAMMA']
    RBF_KERNEL_NUM = config['training']['RBF_KERNEL_NUM']
    HID_DIM = config['training']['HID_DIM']
    NUM_CONV = config['training']['NUM_CONV']
    ATOM_TYPES = config['training']['ATOM_TYPES']
    CUTOFF_DISTANCE1 = config['get_graph']['CUTOFF_DISTANCE1']
    CUTOFF_DISTANCE2 = config['get_graph']['CUTOFF_DISTANCE2']
    EXPONENT = config['training']['EXPONENT']
    Out_model= config['get_potential']['out_model']
    element_list= config['get_potential']['element_list']
    print("\n\t------------------Step--Six---------------------\n")
    print("\tGenerating the 'specified model' parameters for LAMMPS.\n")
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
        print("\n\t ================================================")
        print('\t|      GCNFF is a molecular dynamics force       |')
        print('\t|  field based on Graph Convolutional Networks.  |')
        print("\t ================================================")
        print('\t|            The general use form is :           |')
        print('\t|                                                |')
        print('\t|      gcnff [action commond] [input.json]       |')
        print('\t|                                                |')
        print("\t --------------[action commond]------------------")
        print('\t|                                                |')
        print('\t|                1. get_graph                    |')
        print('\t|                2. divide_set                   |')
        print('\t|                3. init_train                   |')
        print('\t|                4. final_train                  |')
        print('\t|                5. model_test                   |')
        print('\t|                6. get_potential                |')
        print('\t|                                                |')
        print("\t --------------[Accessibility]-------------------")
        print('\t|                                                |')
        print('\t|             Draw the output file               |')
        print('\t|                For example :                   |')
        print('\t|                                                |')
        print('\t|      gcnff showfig pred-ture_energy.txt        |')
        print('\t|                                                |')
        print("\t ================================================")
        exit(-1)
    action = sys.argv[1]
    if action == '-help':
        print("\n\t ================================================")
        print('\t|      GCNFF is a molecular dynamics force       |')
        print('\t|  field based on Graph Convolutional Networks.  |')
        print("\t ================================================")
        print('\t|            The general use form is :           |')
        print('\t|                                                |')
        print('\t|      gcnff [action commond] [input.json]       |')
        print('\t|                                                |')
        print("\t --------------[action commond]------------------")
        print('\t|                                                |')
        print('\t|                1. get_graph                    |')
        print('\t|                2. divide_set                   |')
        print('\t|                3. init_train                   |')
        print('\t|                4. final_train                  |')
        print('\t|                5. model_test                   |')
        print('\t|                6. get_potential                |')
        print('\t|                                                |')
        print("\t --------------[Accessibility]-------------------")
        print('\t|                                                |')
        print('\t|             Draw the output file               |')
        print('\t|                For example :                   |')
        print('\t|                                                |')
        print('\t|      gcnff showfig pred-ture_energy.txt        |')
        print('\t|                                                |')
        print("\t ================================================")
        exit(0)

    if action not in ['get_graph', 'divide_set', 'init_train', 'final_train', 'model_test', 'get_potential','showfig','-help']:
        print("\n\t ================================================")
        print('\t|      GCNFF is a molecular dynamics force       |')
        print('\t|  field based on Graph Convolutional Networks.  |')
        print("\t ================================================")
        print('\t|            The general use form is :           |')
        print('\t|                                                |')
        print('\t|      gcnff [action commond] [input.json]       |')
        print('\t|                                                |')
        print("\t --------------[action commond]------------------")
        print('\t|                                                |')
        print('\t|                1. get_graph                    |')
        print('\t|                2. divide_set                   |')
        print('\t|                3. init_train                   |')
        print('\t|                4. final_train                  |')
        print('\t|                5. model_test                   |')
        print('\t|                6. get_potential                |')
        print('\t|                                                |')
        print("\t --------------[Accessibility]-------------------")
        print('\t|                                                |')
        print('\t|             Draw the output file               |')
        print('\t|                For example :                   |')
        print('\t|                                                |')
        print('\t|      gcnff showfig pred-ture_energy.txt        |')
        print('\t|                                                |')
        print("\t ================================================")
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

    if action == 'model_test':
        model_test(config_file)

    if action == 'get_potential':
        get_potential(config_file)
        
    if action == 'showfig':
        showfig(config_file)
        
if __name__ == '__main__':
    main()