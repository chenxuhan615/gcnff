import json
import torch
import pickle
import numpy as np
from torch_geometric.data import DataLoader
from gcnff.model import GCNFF_init, GCNFF

def load_file(config):
    f = open(config)
    return json.load(f)
    
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
    if(is_pin_memory=="True"):
        test_dataloader = DataLoader(test_graph, batch_size=batch_num, pin_memory=True)
    else:
        test_dataloader = DataLoader(test_graph, batch_size=batch_num, pin_memory=False)
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
