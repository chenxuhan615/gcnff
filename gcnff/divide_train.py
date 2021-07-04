import json
import torch
import pickle
import datetime
import numpy as np
from torch_geometric.data import DataLoader
from gcnff.model import GCNFF_init, global_mean_pool, GCNFF
from gcnff.AutomaticWeightedLoss import AutomaticWeightedLoss

def load_file(config):
    f = open(config)
    return json.load(f)
    
def divide_fix_train(config_file):
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


def divide_free_train(config_file):
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
    
