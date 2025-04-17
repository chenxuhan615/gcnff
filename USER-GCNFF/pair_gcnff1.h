/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
PairStyle(gcnff1,PairGCNFF1)
#else

#ifndef LMP_PAIR_GCNFF1_H
#define LMP_PAIR_GCNFF1_H
#include <torch/torch.h>
#include <vector>
#include "pair.h"

using namespace std;
namespace LAMMPS_NS {

class PairGCNFF1 : public Pair
{
public:
	PairGCNFF1(class LAMMPS *);
	virtual		~PairGCNFF1();
	virtual		void compute(int, int);
	void		settings(int, char **);
	virtual		void coeff(int, char **);
	void		init_style();
	void		init_list(int, class NeighList *);
	void		grab(FILE *, int, double *);
	double		init_one(int, int);
	int			pack_forward_comm(int, int *, double *, int, int *);
	void		unpack_forward_comm(int, int, double *);
	int			pack_reverse_comm(int, int, double *);
	void		unpack_reverse_comm(int, int *, double *);
	double		memory_usage();
	
	struct		Param 
	{
		double	cut;
		int		ielement;
		int		znum;
	};
	
protected:
	double		cutmax;	
	int			nelements;
	int			HID_DIM,RBF_KERNEL_NUM,NUM_CONV,HALF_HID_DIM;
	double		GAMMA,CUTOFF1,CUTOFF2,EXPONENT;

	double		**embedding_weight;
	double		**interaction_blocks_0_linear1_weight;
	double		 *interaction_blocks_0_linear1_bias;
	double		**interaction_blocks_0_filter_block_linear1_weight;
	double		 *interaction_blocks_0_filter_block_linear1_bias;
	double		**interaction_blocks_0_filter_block_linear2_weight;
	double		 *interaction_blocks_0_filter_block_linear2_bias;
	double		**interaction_blocks_0_linear2_weight;
	double		 *interaction_blocks_0_linear2_bias;
	double		**interaction_blocks_0_linear3_weight;
	double		 *interaction_blocks_0_linear3_bias;
	double		**atomwise1_weight;
	double		 *atomwise1_bias;
	double		**atomwise2_weight;
	double		 *atomwise2_bias;
	
	double		**rho_v;
	char		**elements;	
	int			*elem2param;
	int			*map;
	int			nparams;
	Param		*params;

	//virtual void allocate();
	void		read_file(char *);
	virtual		void setup_params();
	int			nmax;
	int			maxNeighbors;
	
	/************Define the GCNFF1 struct****************************/
	struct		Schnet: torch::nn::Module
	{
		Schnet(int atom_type=2,int hidden_layer_dimensions=64,int rbfkernel_number=300)	// default value for the NN
			:embedding(register_module("embedding",torch::nn::Embedding(atom_type,hidden_layer_dimensions)))
			,interaction_blocks_0_linear1				(register_module("interaction_blocks_0_linear1",
															torch::nn::Linear(hidden_layer_dimensions,hidden_layer_dimensions)))
			,interaction_blocks_0_filter_block_linear1	(register_module("interaction_blocks_0_filter_block_linear1",
															torch::nn::Linear(rbfkernel_number,hidden_layer_dimensions)))
			,interaction_blocks_0_filter_block_linear2	(register_module("interaction_blocks_0_filter_block_linear2",
															torch::nn::Linear(hidden_layer_dimensions,hidden_layer_dimensions)))
			,interaction_blocks_0_linear2				(register_module("interaction_blocks_0_linear2",
															torch::nn::Linear(hidden_layer_dimensions,hidden_layer_dimensions)))
			,interaction_blocks_0_linear3				(register_module("interaction_blocks_0_linear3",
															torch::nn::Linear(hidden_layer_dimensions,hidden_layer_dimensions)))
			,atomwise1									(register_module("atomwise1",
															torch::nn::Linear(hidden_layer_dimensions,int(hidden_layer_dimensions/2))))
			,atomwise2									(register_module("atomwise2",
															torch::nn::Linear(int(hidden_layer_dimensions/2),1)))
				{}
		torch::Tensor	forward(torch::Tensor	g_x,torch::Tensor	g_pos,torch::Tensor	g_edge_index1,torch::Tensor	g_edge_index2,
								double cutoff1=6,double cutoff2=3,double gamma=0.5,int rbfkernel_number=300,int hidden_layer_dimensions=64,double exponent=5)
			{
				
				dist1=		torch::index_select(g_pos,0,g_edge_index1[1])-torch::index_select(g_pos,0,g_edge_index1[0]);
				dist1=		torch::norm(dist1,2,1,true);
				rbf_kernel1=	torch::linspace(0.0,cutoff1,rbfkernel_number);
				rbf_tensor1=	dist1-rbf_kernel1;
				rbf_tensor1=	torch::exp(-gamma*torch::mul(rbf_tensor1,rbf_tensor1));
				if(g_edge_index2.size(1)!=0)
				{
					dist2=			torch::index_select(g_pos,0,g_edge_index2[1])-torch::index_select(g_pos,0,g_edge_index2[0]);
					dist2_0=		torch::index_select(g_pos,0,g_edge_index2[0])-g_pos[0];
					dist2_1=		torch::index_select(g_pos,0,g_edge_index2[1])-g_pos[0];
					dist2=			torch::norm(dist2,2,1,true);
					dist2_0=		torch::norm(dist2_0,2,1,true);
					dist2_1=		torch::norm(dist2_1,2,1,true);
					rbf_kernel2=	torch::linspace(0.0,cutoff2,rbfkernel_number);
					rbf_tensor2=	dist2-rbf_kernel2;
					rbf_tensor2=	torch::exp(-gamma*torch::mul(rbf_tensor2,rbf_tensor2));
				}
				temp=		embedding->forward(g_x);
				weight1_0_=	interaction_blocks_0_filter_block_linear1->forward(rbf_tensor1);
				weight1_0_=	torch::log(torch::exp(weight1_0_)+1.0)-torch::log(torch::tensor(2.0));
				weight1_0_=	interaction_blocks_0_filter_block_linear2->forward(weight1_0_);
				weight1_0_=	torch::log(torch::exp(weight1_0_)+1.0)-torch::log(torch::tensor(2.0));
				weight1_0_=	weight1_0_*(1+torch::cos(3.14159265*dist1/cutoff1));
				if(g_edge_index2.size(1)!=0)
				{
					weight2_0_=	interaction_blocks_0_filter_block_linear1->forward(rbf_tensor2);
					weight2_0_=	torch::log(torch::exp(weight2_0_)+1.0)-torch::log(torch::tensor(2.0));
					weight2_0_=	interaction_blocks_0_filter_block_linear2->forward(weight2_0_);
					weight2_0_=	torch::log(torch::exp(weight2_0_)+1.0)-torch::log(torch::tensor(2.0));
					weight2_0_=	weight2_0_*(1+exponent*torch::pow(dist2/cutoff2,exponent+1)-(exponent+1)*torch::pow(dist2/cutoff2,exponent))*
											(1+exponent*torch::pow(dist2_0/cutoff1,exponent+1)-(exponent+1)*torch::pow(dist2_0/cutoff1,exponent))*
											(1+exponent*torch::pow(dist2_1/cutoff1,exponent+1)-(exponent+1)*torch::pow(dist2_1/cutoff1,exponent));
					weight_0_=torch::cat({weight1_0_,weight2_0_});
					g_edge_index=torch::cat({g_edge_index1,g_edge_index2},1);
				}
				else if(g_edge_index2.size(1)==0)
				{
					weight_0_=weight1_0_;
					g_edge_index=g_edge_index1;
				}
				x=		temp.clone().detach();
				temp=	interaction_blocks_0_linear1->forward(temp);
				x_j=torch::gather(temp,0,g_edge_index[1].unsqueeze(1).expand_as(torch::ones({g_edge_index.size(1),temp.size(1)})));
				y_j=x_j*weight_0_;
				tmp_list=torch::zeros({temp.size(0),temp.size(1)}).scatter_add_(0,g_edge_index[0].unsqueeze(1).expand_as(torch::ones({g_edge_index.size(1),temp.size(1)})),y_j);
				temp=tmp_list;
				temp=interaction_blocks_0_linear2->forward(temp);
				temp=torch::log(torch::exp(temp)+1.0)-torch::log(torch::tensor(2.0));
				temp=interaction_blocks_0_linear3->forward(temp);
				temp=torch::add(temp,x);
				temp=atomwise1->forward(temp);
				temp=torch::log(torch::exp(temp)+1.0)-torch::log(torch::tensor(2.0));
				temp=atomwise2->forward(temp);
				return temp;
			}																								// end of tensor forward
		torch::nn::Embedding	embedding;
		torch::nn::Linear		interaction_blocks_0_linear1;
		torch::nn::Linear		interaction_blocks_0_filter_block_linear1;
		torch::nn::Linear		interaction_blocks_0_filter_block_linear2;
		torch::nn::Linear		interaction_blocks_0_linear2;
		torch::nn::Linear		interaction_blocks_0_linear3;
		torch::Tensor			weight1_0_,weight2_0_,weight_0_;
		torch::Tensor			dist1,rbf_kernel1,rbf_tensor1,dist2,dist2_0,dist2_1,rbf_kernel2,rbf_tensor2,temp,x,tmp_list,g_edge_index,x_j,y_j; //
		torch::nn::Linear		atomwise1,atomwise2;
	};																										// end of GCNFF1 structure

	torch::Tensor Embedding_weight;
	torch::Tensor Interaction_blocks_0_linear1_weight;
	torch::Tensor Interaction_blocks_0_linear1_bias;
	torch::Tensor Interaction_blocks_0_filter_block_linear1_weight;
	torch::Tensor Interaction_blocks_0_filter_block_linear1_bias;
	torch::Tensor Interaction_blocks_0_filter_block_linear2_weight;
	torch::Tensor Interaction_blocks_0_filter_block_linear2_bias;
	torch::Tensor Interaction_blocks_0_linear2_weight;
	torch::Tensor Interaction_blocks_0_linear2_bias;
	torch::Tensor Interaction_blocks_0_linear3_weight;
	torch::Tensor Interaction_blocks_0_linear3_bias;
	torch::Tensor Atomwise1_weight;
	torch::Tensor Atomwise1_bias;
	torch::Tensor Atomwise2_weight;
	torch::Tensor Atomwise2_bias;
	/*******************************************************/
	
	virtual		void	allocate();
	torch::Tensor Array2Tensor(std::vector<int>,int);
	torch::Tensor Array2Tensor(int *,int );
	torch::Tensor Array2Tensor(double *,int );
	torch::Tensor Array2Tensor(double **,int ,int );
	torch::Tensor Array2Tensor(int **,int ,int );
};
}
#endif
#endif
/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Cannot open GCNFF1 potential file %s

The specified GCNFF1 potential file cannot be opened.  Check that the path
and name are correct.

E: Incorrect format in GCNFF1 potential file

The potential file is not compatible with the GCNFF1 pair style
implementation in this LAMMPS version.

*/
