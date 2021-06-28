import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import MessagePassing,global_add_pool,global_mean_pool
import warnings
warnings.filterwarnings('ignore')

class RBFLayer(nn.Module):
    def __init__(self,cutoff1=6.0,cutoff2=3.0,gamma=0.1,rbfkernel_number=300):
        super(RBFLayer,self).__init__()
        self.cutoff1=cutoff1
        self.cutoff2=cutoff2
        self.gamma=gamma
        self.rbfkernel_number=rbfkernel_number
        
    def forward(self,g,dist1,dist2,use_device):
        dist1=torch.norm(dist1,p=2,dim=1,keepdim=True)
        dist2=torch.norm(dist2,p=2,dim=1,keepdim=True)
        rbf_kernel1=torch.linspace(0,self.cutoff1,self.rbfkernel_number,dtype=torch.float32,device=use_device)
        rbf_kernel2=torch.linspace(0,self.cutoff2,self.rbfkernel_number,dtype=torch.float32,device=use_device)
        rbf_tensor1=dist1-rbf_kernel1
        rbf_tensor2=dist2-rbf_kernel2
        rbf_tensor1=torch.exp(-self.gamma*torch.mul(rbf_tensor1,rbf_tensor1))
        rbf_tensor2=torch.exp(-self.gamma*torch.mul(rbf_tensor2,rbf_tensor2))
        dist2_0=torch.index_select(dist1,0,g.edge_index2[4])
        dist2_1=torch.index_select(dist1,0,g.edge_index2[5])
        return rbf_tensor1,dist1,rbf_tensor2,dist2,dist2_0,dist2_1


class FilterGeneratorBlock(nn.Module):
    def __init__(self,rbfkernel_number=300,out_dim=64):
        super(FilterGeneratorBlock,self).__init__()
        self.linear1=nn.Linear(rbfkernel_number,out_dim)
        self.linear2=nn.Linear(out_dim,out_dim)
        
    def forward(self,rbf_tensor):
        weight=self.linear1(rbf_tensor)
        weight=torch.log(torch.exp(weight)+1.0)-torch.log(torch.tensor(2.0))
        weight=self.linear2(weight)
        weight=torch.log(torch.exp(weight)+1.0)-torch.log(torch.tensor(2.0))
        return weight


class cfconv(MessagePassing):
    def __init__(self,rbfkernel_number=300,feat_num=64,aggragate='add',node_flow='target_to_source'):
        super(cfconv,self).__init__(aggr=aggragate,flow=node_flow)
        self.filter_block=FilterGeneratorBlock(rbfkernel_number=rbfkernel_number,out_dim=feat_num)
     
    def message(self,x_j,weight):
        return torch.mul(x_j,weight)
    
    def update(self,aggr_out):
        return aggr_out
    
    def forward(self,x1,x1_ghost,edge_index1,edge_index2,rbf_tensor1,dist1,cutoff1,rbf_tensor2,dist2,cutoff2,exponent,dist2_0,dist2_1):
        weight1=self.filter_block(rbf_tensor1)
        weight2=self.filter_block(rbf_tensor2)
        weight1=weight1*(1+torch.cos(3.14159265*dist1/cutoff1))*1.0
        weight2=weight2*(1+exponent*(dist2/cutoff2)**(exponent+1)-(exponent+1)*(dist2/cutoff2)**exponent)*(1+exponent*(dist2_0/cutoff1)**(exponent+1)-(exponent+1)*(dist2_0/cutoff1)**exponent)*(1+exponent*(dist2_1/cutoff1)**(exponent+1)-(exponent+1)*(dist2_1/cutoff1)**exponent)
        weight=torch.cat([weight1,weight2])
        edge_index=torch.cat([edge_index1[[0,2],:],edge_index2[[2,3],:]],1)
        x=torch.cat([x1,x1_ghost])
        weight=torch.cat([weight,weight])
        edge_index=torch.cat([edge_index,edge_index[[1,0],:]],1)
        output=self.propagate(edge_index,x=x,weight=weight)
        return output[:x1.size(0)],output[x1.size(0):]


class InteractionBlock(nn.Module):
    def __init__(self,rbfkernel_number=300,feat_num=64):
        super(InteractionBlock,self).__init__()
        self.linear1=nn.Linear(feat_num,feat_num)
        self.cfconvlayer=cfconv(rbfkernel_number=rbfkernel_number,feat_num=feat_num)
        self.linear2=nn.Linear(feat_num,feat_num)
        self.linear3=nn.Linear(feat_num,feat_num)
    
    def forward(self,edge_index1,edge_index2,node_feature,node_feature_ghost,rbf_tensor1,dist1,rbf_tensor2,dist2,cutoff1,cutoff2,exponent,dist2_0,dist2_1):
        x=torch.tensor(node_feature)
        x_ghost=torch.tensor(node_feature_ghost)
        node_feature=self.linear1(node_feature)
        node_feature_ghost=self.linear1(node_feature_ghost)
        node_feature,node_feature_ghost=self.cfconvlayer(node_feature,node_feature_ghost,edge_index1,edge_index2,rbf_tensor1,dist1,cutoff1,rbf_tensor2,dist2,cutoff2,exponent,dist2_0,dist2_1)
        node_feature=self.linear2(node_feature)
        node_feature_ghost=self.linear2(node_feature_ghost)
        node_feature=torch.log(torch.exp(node_feature)+1.0)-torch.log(torch.tensor(2.0))
        node_feature_ghost=torch.log(torch.exp(node_feature_ghost)+1.0)-torch.log(torch.tensor(2.0))
        node_feature=self.linear3(node_feature)
        node_feature_ghost=self.linear3(node_feature_ghost)
        node_feature=torch.add(node_feature,x)
        node_feature_ghost=torch.add(node_feature_ghost,x_ghost)
        return node_feature,node_feature_ghost


class GCNFF_init(nn.Module):
    def __init__(self,cutoff1=6,cutoff2=3,gamma=0.5,rbfkernel_number=300,
                 hidden_layer_dimensions=64,num_conv=3,atom_types=1,exponent=5):
        super(GCNFF_init,self).__init__()
        self.num_conv=num_conv
        self.cutoff1=cutoff1
        self.cutoff2=cutoff2
        self.exponent=exponent
        
        self.rbf_layer=RBFLayer(cutoff1=cutoff1,cutoff2=cutoff2,gamma=gamma,rbfkernel_number=rbfkernel_number)
        self.embedding=nn.Embedding(atom_types,hidden_layer_dimensions)
        self.interaction_blocks=InteractionBlock(feat_num=hidden_layer_dimensions,
                                                rbfkernel_number=rbfkernel_number)
        self.atomwise1=nn.Linear(hidden_layer_dimensions,int(hidden_layer_dimensions/2))
        self.atomwise2=nn.Linear(int(hidden_layer_dimensions/2),1)
        
    def forward(self,g,dist1,dist2,use_device):
        rbf_tensor1,dist1,rbf_tensor2,dist2,dist2_0,dist2_1=self.rbf_layer(g,dist1,dist2,use_device)
        temp=self.embedding(g.x)
        temp_ghost=self.embedding(g.x_ghost)
        for i in range(self.num_conv):
            temp,temp_ghost=self.interaction_blocks(g.edge_index1,g.edge_index2,temp,temp_ghost,rbf_tensor1,dist1,rbf_tensor2,dist2,self.cutoff1,self.cutoff2,self.exponent,dist2_0,dist2_1)
        temp=self.atomwise1(temp)
        temp=torch.log(torch.exp(temp)+1.0)-torch.log(torch.tensor(2.0))
        temp=self.atomwise2(temp)
        temp=global_add_pool(temp,g.batch)
        return temp


class GCNFF(nn.Module):
    def __init__(self,cutoff1=6,cutoff2=3,gamma=0.5,rbfkernel_number=300,
                 hidden_layer_dimensions=64,num_conv=3,atom_types=1,exponent=5):
        super(GCNFF,self).__init__()
        self.num_conv=num_conv
        self.cutoff1=cutoff1
        self.cutoff2=cutoff2
        self.exponent=exponent
        
        self.rbf_layer=RBFLayer(cutoff1=cutoff1,cutoff2=cutoff2,gamma=gamma,rbfkernel_number=rbfkernel_number)
        self.embedding=nn.Embedding(atom_types,hidden_layer_dimensions)
        self.interaction_blocks=nn.ModuleList([InteractionBlock(feat_num=hidden_layer_dimensions,
                                                                rbfkernel_number=rbfkernel_number) 
                                               for i in range(num_conv)])
        self.atomwise1=nn.Linear(hidden_layer_dimensions,int(hidden_layer_dimensions/2))
        self.atomwise2=nn.Linear(int(hidden_layer_dimensions/2),1)
        
    def forward(self,g,dist1,dist2,use_device):
        rbf_tensor1,dist1,rbf_tensor2,dist2,dist2_0,dist2_1=self.rbf_layer(g,dist1,dist2,use_device)
        temp=self.embedding(g.x)
        temp_ghost=self.embedding(g.x_ghost)
        for i in range(self.num_conv):
            temp,temp_ghost=self.interaction_blocks[i](g.edge_index1,g.edge_index2,temp,temp_ghost,rbf_tensor1,dist1,rbf_tensor2,dist2,self.cutoff1,self.cutoff2,self.exponent,dist2_0,dist2_1)
        temp=self.atomwise1(temp)
        temp=torch.log(torch.exp(temp)+1.0)-torch.log(torch.tensor(2.0))
        temp=self.atomwise2(temp)
        temp=global_add_pool(temp,g.batch)
        return temp