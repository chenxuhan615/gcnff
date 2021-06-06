import math
import torch
import numpy as np
from torch_geometric.data import Data

def getatom_num(filename):
    with open(filename,'r') as f:
        total_atoms,atom_num,total_config=f.readline().split()
        total_atoms=int(total_atoms)
        atom_num=int(atom_num)
        total_config=int(total_config)
        return atom_num

def generate_reps(cutoff_dist,vec_1_len,vec_2_len,vec_3_len):
    vec_1_num=math.ceil(cutoff_dist/vec_1_len)
    vec_2_num=math.ceil(cutoff_dist/vec_2_len)
    vec_3_num=math.ceil(cutoff_dist/vec_3_len)
    vec_1_iter=[0]
    vec_2_iter=[0]
    vec_3_iter=[0]
    for i in range(vec_1_num):
        vec_1_iter.append(i+1)
        vec_1_iter.append(-i-1)
    for j in range(vec_2_num):
        vec_2_iter.append(j+1)
        vec_2_iter.append(-j-1)
    for k in range(vec_3_num):
        vec_3_iter.append(k+1)
        vec_3_iter.append(-k-1)
    return vec_1_iter,vec_2_iter,vec_3_iter

def generate_graphs1(filename,graph_list,cutoff_dist1,cutoff_dist2,jumpNum,baseEvalue):
    with open(filename,'r') as f:
        total_atoms,atom_num,total_config=f.readline().split()
        total_atoms=int(total_atoms)
        atom_num=int(atom_num)
        total_config=int(total_config)
        for i in range(total_config):
            if i%jumpNum!=0:
                for j in range(atom_num+4): 
                    f.readline()
                continue
            vec_1_x,vec_1_y,vec_1_z=f.readline().split()
            vec_1_x=float(vec_1_x)
            vec_1_y=float(vec_1_y)
            vec_1_z=float(vec_1_z)
            vec_2_x,vec_2_y,vec_2_z=f.readline().split()
            vec_2_x=float(vec_2_x)
            vec_2_y=float(vec_2_y)
            vec_2_z=float(vec_2_z)
            vec_3_x,vec_3_y,vec_3_z=f.readline().split()
            vec_3_x=float(vec_3_x)
            vec_3_y=float(vec_3_y)
            vec_3_z=float(vec_3_z)
            vec_1_len=math.sqrt(vec_1_x*vec_1_x+vec_1_y*vec_1_y+vec_1_z*vec_1_z)
            vec_2_len=math.sqrt(vec_2_x*vec_2_x+vec_2_y*vec_2_y+vec_2_z*vec_2_z)
            vec_3_len=math.sqrt(vec_3_x*vec_3_x+vec_3_y*vec_3_y+vec_3_z*vec_3_z)
            vec_1_iter,vec_2_iter,vec_3_iter=generate_reps(cutoff_dist1,vec_1_len,vec_2_len,vec_3_len)
            len_vec1=len(vec_1_iter)
            len_vec2=len(vec_2_iter)
            len_vec3=len(vec_3_iter)
            
            x_list=list()
            y_list=list()
            z_list=list()
            fx_list=list()
            fy_list=list()
            fz_list=list()
            charge_list=list()
            edge_list_u1=list()
            edge_list_v1=list()
            edge_list_x1=list()
            period_vec_list1=list()
            edge_list_u2=list()
            edge_list_v2=list()
            edge_list_w2=list()
            edge_list_x2=list()
            edge_list_y2=list()
            period_vec_list2=list()
            
            for j in range(atom_num):
                atom_type,x,y,z,fx,fy,fz=f.readline().split()
                atom_type=int(atom_type)
                x_list.append(float(x))
                y_list.append(float(y))
                z_list.append(float(z))
                fx_list.append(float(fx))
                fy_list.append(float(fy))
                fz_list.append(float(fz))
                charge_list.append(atom_type-1)
            total_energy=f.readline()
            total_energy=float(total_energy)
            
            if total_energy/atom_num>baseEvalue:
                continue

            ghost_atom=0
            charge_ghost_list=list()

            for m in range(atom_num):
                x_list_out=list()
                y_list_out=list()
                z_list_out=list()
                true_index=list()
                ghost_index=list()
                x_list_out.append(x_list[m])
                y_list_out.append(y_list[m])
                z_list_out.append(z_list[m])
                true_index.append(m)
                ghost_index.append(m)
                for n in range(atom_num):
                    if n!=m:
                        for period_num in range(len_vec1*len_vec2*len_vec3):
                            temp=period_num
                            vec_1_period=temp%len_vec1
                            temp//=len_vec1
                            vec_2_period=temp%len_vec2
                            vec_3_period=temp//len_vec2
                        
                            x_dist=x_list[m]-x_list[n]-vec_1_iter[vec_1_period]*vec_1_x-vec_2_iter[vec_2_period]*vec_2_x-vec_3_iter[vec_3_period]*vec_3_x
                            y_dist=y_list[m]-y_list[n]-vec_1_iter[vec_1_period]*vec_1_y-vec_2_iter[vec_2_period]*vec_2_y-vec_3_iter[vec_3_period]*vec_3_y
                            z_dist=z_list[m]-z_list[n]-vec_1_iter[vec_1_period]*vec_1_z-vec_2_iter[vec_2_period]*vec_2_z-vec_3_iter[vec_3_period]*vec_3_z
                        
                            if x_dist*x_dist+y_dist*y_dist+z_dist*z_dist<cutoff_dist1*cutoff_dist1:
                                charge_ghost_list.append(charge_list[n])
                                x_list_out.append(x_list[n]+vec_1_iter[vec_1_period]*vec_1_x+vec_2_iter[vec_2_period]*vec_2_x+vec_3_iter[vec_3_period]*vec_3_x)
                                y_list_out.append(y_list[n]+vec_1_iter[vec_1_period]*vec_1_y+vec_2_iter[vec_2_period]*vec_2_y+vec_3_iter[vec_3_period]*vec_3_y)
                                z_list_out.append(z_list[n]+vec_1_iter[vec_1_period]*vec_1_z+vec_2_iter[vec_2_period]*vec_2_z+vec_3_iter[vec_3_period]*vec_3_z)
                                true_index.append(n)
                                ghost_index.append(ghost_atom+atom_num)
                                ghost_atom=ghost_atom+1
                                
                for period_num in range(len_vec1*len_vec2*len_vec3):
                        if period_num==0:
                            continue
                            
                        temp=period_num
                        vec_1_period=temp%len_vec1
                        temp//=len_vec1
                        vec_2_period=temp%len_vec2
                        vec_3_period=temp//len_vec2
              
                        x_dist=-vec_1_iter[vec_1_period]*vec_1_x-vec_2_iter[vec_2_period]*vec_2_x-vec_3_iter[vec_3_period]*vec_3_x
                        y_dist=-vec_1_iter[vec_1_period]*vec_1_y-vec_2_iter[vec_2_period]*vec_2_y-vec_3_iter[vec_3_period]*vec_3_y
                        z_dist=-vec_1_iter[vec_1_period]*vec_1_z-vec_2_iter[vec_2_period]*vec_2_z-vec_3_iter[vec_3_period]*vec_3_z
                        
                        if x_dist*x_dist+y_dist*y_dist+z_dist*z_dist<cutoff_dist1*cutoff_dist1:
                            charge_ghost_list.append(charge_list[m])
                            x_list_out.append(x_list[m]+vec_1_iter[vec_1_period]*vec_1_x+vec_2_iter[vec_2_period]*vec_2_x+vec_3_iter[vec_3_period]*vec_3_x)
                            y_list_out.append(y_list[m]+vec_1_iter[vec_1_period]*vec_1_y+vec_2_iter[vec_2_period]*vec_2_y+vec_3_iter[vec_3_period]*vec_3_y)
                            z_list_out.append(z_list[m]+vec_1_iter[vec_1_period]*vec_1_z+vec_2_iter[vec_2_period]*vec_2_z+vec_3_iter[vec_3_period]*vec_3_z)
                            true_index.append(m)
                            ghost_index.append(ghost_atom+atom_num)
                            ghost_atom=ghost_atom+1

                for m1 in range(len(x_list_out)):
                    for n1 in range(m1):
                        x_dist=x_list_out[m1]-x_list_out[n1]
                        y_dist=y_list_out[m1]-y_list_out[n1]
                        z_dist=z_list_out[m1]-z_list_out[n1]
                        if ghost_index[n1]<atom_num:
                            if x_dist*x_dist+y_dist*y_dist+z_dist*z_dist<cutoff_dist1*cutoff_dist1:
                                edge_list_u1.append(m)
                                edge_list_v1.append(true_index[m1])
                                edge_list_x1.append(ghost_index[m1])
                                period_vec_list1.append([x_dist-x_list[true_index[m1]]+x_list[m],
                                                         y_dist-y_list[true_index[m1]]+y_list[m],
                                                         z_dist-z_list[true_index[m1]]+z_list[m]])
                        else:
                            if x_dist*x_dist+y_dist*y_dist+z_dist*z_dist<cutoff_dist2*cutoff_dist2:
                                edge_list_u2.append(true_index[n1])
                                edge_list_v2.append(true_index[m1])
                                edge_list_w2.append(ghost_index[n1])
                                edge_list_x2.append(ghost_index[m1])
                                edge_list_y2.append(m)
                                period_vec_list2.append([x_dist-x_list[true_index[m1]]+x_list[true_index[n1]],
                                                         y_dist-y_list[true_index[m1]]+y_list[true_index[n1]],
                                                         z_dist-z_list[true_index[m1]]+z_list[true_index[n1]]])
                    
            direct_index=np.array([edge_list_u1,edge_list_x1])
            edge_fcut=np.array([edge_list_w2,edge_list_x2,edge_list_y2])
            dist_index1=list()
            dist_index2=list()
            for atom in range(len(edge_list_y2)):
                dist_index1.append(int(np.where((direct_index[0]==edge_fcut[2][atom])&(direct_index[1]==edge_fcut[0][atom]))[0]))
                dist_index2.append(int(np.where((direct_index[0]==edge_fcut[2][atom])&(direct_index[1]==edge_fcut[1][atom]))[0]))

            data_temp=Data(x=torch.tensor(charge_list,dtype=torch.long),
                           x_ghost=torch.tensor(charge_ghost_list,dtype=torch.long),
                           pos=torch.tensor(np.array([x_list,y_list,z_list]).transpose(1,0),dtype=torch.float32,requires_grad=True),
                           force=torch.tensor(np.array([fx_list,fy_list,fz_list]).transpose(1,0),dtype=torch.float32,requires_grad=True),
                           y=torch.tensor([[total_energy]]),
                           edge_index1=torch.tensor([edge_list_u1,edge_list_v1,edge_list_x1],dtype=torch.long),
                           edge_index2=torch.tensor([edge_list_u2,edge_list_v2,edge_list_w2,edge_list_x2,dist_index1,dist_index2],dtype=torch.long),
                           edge_attr1=torch.tensor(period_vec_list1,dtype=torch.float32),
                           edge_attr2=torch.tensor(period_vec_list2,dtype=torch.float32)
                          )
            graph_list.append(data_temp)
            
def generate_graphs2(filename,graph_list,cutoff_dist1,cutoff_dist2,jumpNum,baseEvalue):
    with open(filename,'r') as f:
        total_atoms,atom_num,total_config=f.readline().split()
        total_atoms=int(total_atoms)
        atom_num=int(atom_num)
        total_config=int(total_config)
        
        for i in range(total_config):
            if i%jumpNum!=0:
                for j in range(atom_num+5): 
                    f.readline()
                continue
                
            vec_1_x,vec_1_y,vec_1_z=f.readline().split()
            vec_1_x=float(vec_1_x)
            vec_1_y=float(vec_1_y)
            vec_1_z=float(vec_1_z)
            vec_2_x,vec_2_y,vec_2_z=f.readline().split()
            vec_2_x=float(vec_2_x)
            vec_2_y=float(vec_2_y)
            vec_2_z=float(vec_2_z)
            vec_3_x,vec_3_y,vec_3_z=f.readline().split()
            vec_3_x=float(vec_3_x)
            vec_3_y=float(vec_3_y)
            vec_3_z=float(vec_3_z)
            vec_1_len=math.sqrt(vec_1_x*vec_1_x+vec_1_y*vec_1_y+vec_1_z*vec_1_z)
            vec_2_len=math.sqrt(vec_2_x*vec_2_x+vec_2_y*vec_2_y+vec_2_z*vec_2_z)
            vec_3_len=math.sqrt(vec_3_x*vec_3_x+vec_3_y*vec_3_y+vec_3_z*vec_3_z)
            vec_1_iter,vec_2_iter,vec_3_iter=generate_reps(cutoff_dist1,vec_1_len,vec_2_len,vec_3_len)
            len_vec1=len(vec_1_iter)
            len_vec2=len(vec_2_iter)
            len_vec3=len(vec_3_iter)
            
            x_list=list()
            y_list=list()
            z_list=list()
            fx_list=list()
            fy_list=list()
            fz_list=list()
            charge_list=list()
            edge_list_u1=list()
            edge_list_v1=list()
            edge_list_x1=list()
            period_vec_list1=list()
            edge_list_u2=list()
            edge_list_v2=list()
            edge_list_w2=list()
            edge_list_x2=list()
            edge_list_y2=list()
            period_vec_list2=list()
            
            for j in range(atom_num):
                atom_type,x,y,z,fx,fy,fz=f.readline().split()
                atom_type=int(atom_type)
                x_list.append(float(x))
                y_list.append(float(y))
                z_list.append(float(z))
                fx_list.append(float(fx))
                fy_list.append(float(fy))
                fz_list.append(float(fz))
                charge_list.append(atom_type-1)
            total_energy=f.readline()
            total_energy=float(total_energy)

            v11,v22,v33,v12,v13,v23=f.readline().split()
            v11=float(v11)
            v22=float(v22)
            v33=float(v33)
            v12=float(v12)
            v13=float(v13)
            v23=float(v23)
            
            if total_energy/atom_num>baseEvalue:
                continue

            ghost_atom=0
            charge_ghost_list=list()

            for m in range(atom_num):
                x_list_out=list()
                y_list_out=list()
                z_list_out=list()
                true_index=list()
                ghost_index=list()
                x_list_out.append(x_list[m])
                y_list_out.append(y_list[m])
                z_list_out.append(z_list[m])
                true_index.append(m)
                ghost_index.append(m)
                for n in range(atom_num):
                    if n!=m:
                        for period_num in range(len_vec1*len_vec2*len_vec3):
                            temp=period_num
                            vec_1_period=temp%len_vec1
                            temp//=len_vec1
                            vec_2_period=temp%len_vec2
                            vec_3_period=temp//len_vec2
                        
                            x_dist=x_list[m]-x_list[n]-vec_1_iter[vec_1_period]*vec_1_x-vec_2_iter[vec_2_period]*vec_2_x-vec_3_iter[vec_3_period]*vec_3_x
                            y_dist=y_list[m]-y_list[n]-vec_1_iter[vec_1_period]*vec_1_y-vec_2_iter[vec_2_period]*vec_2_y-vec_3_iter[vec_3_period]*vec_3_y
                            z_dist=z_list[m]-z_list[n]-vec_1_iter[vec_1_period]*vec_1_z-vec_2_iter[vec_2_period]*vec_2_z-vec_3_iter[vec_3_period]*vec_3_z
                        
                            if x_dist*x_dist+y_dist*y_dist+z_dist*z_dist<cutoff_dist1*cutoff_dist1:
                                charge_ghost_list.append(charge_list[n])
                                x_list_out.append(x_list[n]+vec_1_iter[vec_1_period]*vec_1_x+vec_2_iter[vec_2_period]*vec_2_x+vec_3_iter[vec_3_period]*vec_3_x)
                                y_list_out.append(y_list[n]+vec_1_iter[vec_1_period]*vec_1_y+vec_2_iter[vec_2_period]*vec_2_y+vec_3_iter[vec_3_period]*vec_3_y)
                                z_list_out.append(z_list[n]+vec_1_iter[vec_1_period]*vec_1_z+vec_2_iter[vec_2_period]*vec_2_z+vec_3_iter[vec_3_period]*vec_3_z)
                                true_index.append(n)
                                ghost_index.append(ghost_atom+atom_num)
                                ghost_atom=ghost_atom+1
                                
                for period_num in range(len_vec1*len_vec2*len_vec3):
                        if period_num==0:
                            continue
                            
                        temp=period_num
                        vec_1_period=temp%len_vec1
                        temp//=len_vec1
                        vec_2_period=temp%len_vec2
                        vec_3_period=temp//len_vec2
              
                        x_dist=-vec_1_iter[vec_1_period]*vec_1_x-vec_2_iter[vec_2_period]*vec_2_x-vec_3_iter[vec_3_period]*vec_3_x
                        y_dist=-vec_1_iter[vec_1_period]*vec_1_y-vec_2_iter[vec_2_period]*vec_2_y-vec_3_iter[vec_3_period]*vec_3_y
                        z_dist=-vec_1_iter[vec_1_period]*vec_1_z-vec_2_iter[vec_2_period]*vec_2_z-vec_3_iter[vec_3_period]*vec_3_z
                        
                        if x_dist*x_dist+y_dist*y_dist+z_dist*z_dist<cutoff_dist1*cutoff_dist1:
                            charge_ghost_list.append(charge_list[m])
                            x_list_out.append(x_list[m]+vec_1_iter[vec_1_period]*vec_1_x+vec_2_iter[vec_2_period]*vec_2_x+vec_3_iter[vec_3_period]*vec_3_x)
                            y_list_out.append(y_list[m]+vec_1_iter[vec_1_period]*vec_1_y+vec_2_iter[vec_2_period]*vec_2_y+vec_3_iter[vec_3_period]*vec_3_y)
                            z_list_out.append(z_list[m]+vec_1_iter[vec_1_period]*vec_1_z+vec_2_iter[vec_2_period]*vec_2_z+vec_3_iter[vec_3_period]*vec_3_z)
                            true_index.append(m)
                            ghost_index.append(ghost_atom+atom_num)
                            ghost_atom=ghost_atom+1

                for m1 in range(len(x_list_out)):
                    for n1 in range(m1):
                        
                        x_dist=x_list_out[m1]-x_list_out[n1]
                        y_dist=y_list_out[m1]-y_list_out[n1]
                        z_dist=z_list_out[m1]-z_list_out[n1]
                        
                        if ghost_index[n1]<atom_num:
                            if x_dist*x_dist+y_dist*y_dist+z_dist*z_dist<cutoff_dist1*cutoff_dist1:
                                edge_list_u1.append(m)
                                edge_list_v1.append(true_index[m1])
                                edge_list_x1.append(ghost_index[m1])
                                period_vec_list1.append([x_dist-x_list[true_index[m1]]+x_list[m],
                                                         y_dist-y_list[true_index[m1]]+y_list[m],
                                                         z_dist-z_list[true_index[m1]]+z_list[m]])
                        else:
                            if x_dist*x_dist+y_dist*y_dist+z_dist*z_dist<cutoff_dist2*cutoff_dist2:
                                edge_list_u2.append(true_index[n1])
                                edge_list_v2.append(true_index[m1])
                                edge_list_w2.append(ghost_index[n1])
                                edge_list_x2.append(ghost_index[m1])
                                edge_list_y2.append(m)
                                period_vec_list2.append([x_dist-x_list[true_index[m1]]+x_list[true_index[n1]],
                                                         y_dist-y_list[true_index[m1]]+y_list[true_index[n1]],
                                                         z_dist-z_list[true_index[m1]]+z_list[true_index[n1]]])
                    
            direct_index=np.array([edge_list_u1,edge_list_x1])
            edge_fcut=np.array([edge_list_w2,edge_list_x2,edge_list_y2])
            dist_index1=list()
            dist_index2=list()
            for atom in range(len(edge_list_y2)):
                dist_index1.append(int(np.where((direct_index[0]==edge_fcut[2][atom])&(direct_index[1]==edge_fcut[0][atom]))[0]))
                dist_index2.append(int(np.where((direct_index[0]==edge_fcut[2][atom])&(direct_index[1]==edge_fcut[1][atom]))[0]))

            data_temp=Data(x=torch.tensor(charge_list,dtype=torch.long),
                           x_ghost=torch.tensor(charge_ghost_list,dtype=torch.long),
                           pos=torch.tensor(np.array([x_list,y_list,z_list]).transpose(1,0),dtype=torch.float32,requires_grad=True),
                           force=torch.tensor(np.array([fx_list,fy_list,fz_list]).transpose(1,0),dtype=torch.float32,requires_grad=True),
                           y=torch.tensor([[total_energy]]),
                           virial=torch.tensor([[v11,v22,v33,v12,v13,v23]]),
                           edge_index1=torch.tensor([edge_list_u1,edge_list_v1,edge_list_x1],dtype=torch.long),
                           edge_index2=torch.tensor([edge_list_u2,edge_list_v2,edge_list_w2,edge_list_x2,dist_index1,dist_index2],dtype=torch.long),
                           edge_attr1=torch.tensor(period_vec_list1,dtype=torch.float32),
                           edge_attr2=torch.tensor(period_vec_list2,dtype=torch.float32)
                          )
            graph_list.append(data_temp)