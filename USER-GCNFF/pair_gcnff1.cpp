/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: CXH,YY
------------------------------------------------------------------------- */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pair_gcnff1.h"
#include "atom.h"
#include "domain.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "force.h"
#include "comm.h"
#include "memory.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "memory.h"
#include "error.h"
#include "citeme.h"
#include "math_special.h"
#include "math_const.h"
#include <iostream>
#include "update.h"
#include <torch/torch.h>
#include <vector>

using namespace std;
using namespace LAMMPS_NS;
using namespace MathSpecial;

static const char cite_pair_gcnff1[] =
  "pair gcnff1 command:\n\n"
  "@article{botuXXXadaptive,\n"
  " author    = {Chenxu Han, Yang Yang, Hongxiang Zong, XiangDong Ding},\n"
  " title     = {GCNFF: A molecular dynamics force field developed by graph convolution neural network},\n"
  " journal   = {XXXjournal},\n"
  " volume    = {XXXvolume},\n"
  " number    = {XXXnumber},\n"
  " pages     = {XXX--XXXpages},\n"
  " year      = {XXXyear},\n"
  " publisher = {XXXpublisher}\n"
  "}\n\n";

#define GCNFF1_VERSION 3
#define MAXLINE 2048000
#define MAXWORD 1024000

/* ---------------------------------------------------------------------- */

PairGCNFF1::PairGCNFF1(LAMMPS *lmp) : Pair(lmp)
{
  if (lmp->citeme) lmp->citeme->add(cite_pair_gcnff1);
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  //no_virial_fdotr_compute = 1;
  nelements = 0;
  elements = NULL;
  elem2param = NULL;
  nparams = 0;
  params = NULL;
  map = NULL;
  rho_v=NULL;
  //vv = NULL;
  nmax = 0;
  maxNeighbors = 0;
  comm_forward = 8;
  comm_reverse = 0;
  /********Torch************/
  embedding_weight                                 = NULL;
  interaction_blocks_0_linear1_weight              = NULL;
  interaction_blocks_0_linear1_bias                = NULL;
  interaction_blocks_0_filter_block_linear1_weight = NULL;
  interaction_blocks_0_filter_block_linear1_bias   = NULL;
  interaction_blocks_0_filter_block_linear2_weight = NULL;
  interaction_blocks_0_filter_block_linear2_bias   = NULL;
  interaction_blocks_0_linear2_weight              = NULL;
  interaction_blocks_0_linear2_bias                = NULL;
  interaction_blocks_0_linear3_weight              = NULL;
  interaction_blocks_0_linear3_bias                = NULL;
  atomwise1_weight                                 = NULL;
  atomwise1_bias                                   = NULL;
  atomwise2_weight                                 = NULL;
  atomwise2_bias                                   = NULL;
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
   ------------------------------------------------------------------------- */

PairGCNFF1::~PairGCNFF1()
{
  //printf("Free: start"); 
  if (elements)
    for (int i = 0; i < nelements; i++) delete [] elements[i];
  delete [] elements;

  if (params)
  {
    memory->destroy(params);
    params = NULL;
  }
  memory->destroy(rho_v);

  memory->destroy(embedding_weight);
  memory->destroy(interaction_blocks_0_linear1_weight);
  memory->destroy(interaction_blocks_0_linear1_bias);
  memory->destroy(interaction_blocks_0_filter_block_linear1_weight);
  memory->destroy(interaction_blocks_0_filter_block_linear1_bias);
  memory->destroy(interaction_blocks_0_filter_block_linear2_weight);
  memory->destroy(interaction_blocks_0_filter_block_linear2_bias);
  memory->destroy(interaction_blocks_0_linear2_weight);
  memory->destroy(interaction_blocks_0_linear2_bias);
  memory->destroy(interaction_blocks_0_linear3_weight);
  memory->destroy(interaction_blocks_0_linear3_bias);
  memory->destroy(atomwise1_weight);
  memory->destroy(atomwise1_bias);
  memory->destroy(atomwise2_weight);
  memory->destroy(atomwise2_bias);

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    delete [] map;
  }
}

/* ---------------------------------------------------------------------- */

void PairGCNFF1::compute(int eflag, int vflag)
{
	int		i,j,k,ii,jj,inum,jnum,itype,jtype;
	double	xtmp,ytmp,ztmp,delx,dely,delz,rsq;
	double  delx_1,dely_1,delz_1,rsq_1, delx_2,dely_2,delz_2,rsq_2;
	int		 *type,*ilist,*jlist,*numneigh,**firstneigh;
	//ev_init(eflag,vflag);
	if	(eflag || vflag)
		ev_setup(eflag,vflag);
	else	
		evflag = vflag_fdotr =  eflag_global = vflag_global = eflag_atom = vflag_atom = 0;
	if(atom->nmax > nmax)
	{
		memory->destroy(rho_v);
		nmax = atom->nmax;
		memory->create(rho_v,nmax,8,"pair:rho_v");
	}
	double		**x = atom->x;											//position
	double		**f = atom->f;											//force
	int			nlocal = atom->nlocal;
	bool		newton_pair = force->newton_pair;
	type = atom->type;
	inum = listfull->inum;
	ilist = listfull->ilist;
	numneigh = listfull->numneigh;
	firstneigh = listfull->firstneigh;

	int *edge_num;
	memory->create(edge_num,inum,"pair:edge_num");
	for(ii = 0; ii < inum; ii++)
	{
		i = ilist[ii];
		jnum = numneigh[i];
		jlist = firstneigh[i];
		edge_num[i]=0;
		for(jj=0;jj<jnum;jj++)
		{
			j = jlist[jj];
			j &= NEIGHMASK;
			delx = x[j][0] - x[i][0];
			dely = x[j][1] - x[i][1];
			delz = x[j][2] - x[i][2];
			 rsq = delx*delx + dely*dely + delz*delz;
			if(rsq>0&&rsq<CUTOFF1*CUTOFF1)
			{
				edge_num[i] += 1;
			}
		}
	}
	/************Define the GCNFF1 and initiolization**********/
	Schnet schnet(nparams,HID_DIM,RBF_KERNEL_NUM);
	schnet.embedding->weight=Embedding_weight;
	schnet.interaction_blocks_0_linear1->weight					=Interaction_blocks_0_linear1_weight;
	schnet.interaction_blocks_0_linear1->bias					=Interaction_blocks_0_linear1_bias;
	schnet.interaction_blocks_0_filter_block_linear1->weight	=Interaction_blocks_0_filter_block_linear1_weight;
	schnet.interaction_blocks_0_filter_block_linear1->bias		=Interaction_blocks_0_filter_block_linear1_bias;
	schnet.interaction_blocks_0_filter_block_linear2->weight	=Interaction_blocks_0_filter_block_linear2_weight;
	schnet.interaction_blocks_0_filter_block_linear2->bias		=Interaction_blocks_0_filter_block_linear2_bias;
	schnet.interaction_blocks_0_linear2->weight					=Interaction_blocks_0_linear2_weight;
	schnet.interaction_blocks_0_linear2->bias					=Interaction_blocks_0_linear2_bias;
	schnet.interaction_blocks_0_linear3->weight					=Interaction_blocks_0_linear3_weight;
	schnet.interaction_blocks_0_linear3->bias					=Interaction_blocks_0_linear3_bias;
	schnet.atomwise1->weight									=Atomwise1_weight;
	schnet.atomwise1->bias										=Atomwise1_bias;
	schnet.atomwise2->weight									=Atomwise2_weight;
	schnet.atomwise2->bias										=Atomwise2_bias;
	/********************************************************/
	
	for (ii = 0; ii < inum; ii++)
	{
		int E_num=0;
		i = ilist[ii];
		itype = map[type[i]];
		xtmp = x[i][0];
		ytmp = x[i][1];
		ztmp = x[i][2];
		const Param &iparam = params[elem2param[itype]];
		jlist = firstneigh[i];
		jnum = numneigh[i];
		int *ztype_a;
		double **x_a;
		int *graphmap;
		memory->create(ztype_a,edge_num[i]+1,"pair:ztype_a");
		memory->create(x_a,edge_num[i]+1,3,"pair:x_a");
		memory->create(graphmap,edge_num[i],"pair:graphmap");
		ztype_a[0]=iparam.znum;
		x_a[0][0]=xtmp;
		x_a[0][1]=ytmp;
		x_a[0][2]=ztmp;
		vector<int> edge_index1_0;
		vector<int> edge_index1_1;
		vector<int> edge_index2_0;
		vector<int> edge_index2_1;
		for (jj = 0; jj < jnum; jj++) 
		{
			j = jlist[jj];
			j &= NEIGHMASK;
			jtype = map[type[j]];
			delx = x[j][0] - xtmp;
			dely = x[j][1] - ytmp;
			delz = x[j][2] - ztmp;
			rsq = delx*delx + dely*dely + delz*delz;
			if((rsq > 0.0) && (rsq < CUTOFF1*CUTOFF1))
			{
				graphmap[E_num]=j;
				E_num++;
				const Param &jparam = params[elem2param[jtype]];
				ztype_a[E_num]=jparam.znum;
				x_a[E_num][0]=x[j][0];
				x_a[E_num][1]=x[j][1];
				x_a[E_num][2]=x[j][2];
				edge_index1_0.push_back(E_num);
				edge_index1_1.push_back(0);
				edge_index1_0.push_back(0);
				edge_index1_1.push_back(E_num);
			}
		}
		
		for(int atom1=0;atom1<(edge_num[i]+1);atom1++)
			for(int atom2=1;atom2<atom1;atom2++)
			{
				delx_1=x_a[atom1][0]-x_a[0][0];
				dely_1=x_a[atom1][1]-x_a[0][1];
				delz_1=x_a[atom1][2]-x_a[0][2];
				rsq_1 = delx_1*delx_1 + dely_1*dely_1 + delz_1*delz_1;
				if (rsq_1 < CUTOFF2 * CUTOFF2)
				{
					delx_2=x_a[atom2][0]-x_a[0][0];
					dely_2=x_a[atom2][1]-x_a[0][1];
					delz_2=x_a[atom2][2]-x_a[0][2];
					rsq_2 = delx_2*delx_2 + dely_2*dely_2 + delz_2*delz_2;
					if (rsq_2 < CUTOFF2 * CUTOFF2)
					{
						delx=x_a[atom1][0]-x_a[atom2][0];
						dely=x_a[atom1][1]-x_a[atom2][1];
						delz=x_a[atom1][2]-x_a[atom2][2];
						rsq = delx*delx + dely*dely + delz*delz;
						if(rsq < CUTOFF2*CUTOFF2)
						{
							edge_index2_0.push_back(atom1);
							edge_index2_1.push_back(atom2);
							edge_index2_0.push_back(atom2);
							edge_index2_1.push_back(atom1);
						}
					}
				}
			}
			
		/***********Transform into TENSOR format****************/
		torch::Tensor	edge_index1_0_t		=	Array2Tensor(edge_index1_0,edge_index1_0.size());
		vector<int>().swap(edge_index1_0);
		torch::Tensor	edge_index1_1_t		=	Array2Tensor(edge_index1_1,edge_index1_1.size());
		vector<int>().swap(edge_index1_1);
		torch::Tensor edge_index1_t	=	torch::cat({edge_index1_0_t.unsqueeze(0),edge_index1_1_t.unsqueeze(0)});
		
		torch::Tensor	edge_index2_0_t		=	Array2Tensor(edge_index2_0,edge_index2_0.size());
		vector<int>().swap(edge_index2_0);
		torch::Tensor	edge_index2_1_t		=	Array2Tensor(edge_index2_1,edge_index2_1.size());
		vector<int>().swap(edge_index2_1);
		torch::Tensor edge_index2_t	=	torch::cat({edge_index2_0_t.unsqueeze(0),edge_index2_1_t.unsqueeze(0)});
		
		torch::Tensor ztype_t		=	Array2Tensor(ztype_a,edge_num[i]+1);
		torch::Tensor x_t			=	Array2Tensor(x_a,edge_num[i]+1,3);
		memory->destroy(ztype_a);
		memory->destroy(x_a);
		/*********Predict the Energy and Force******************/
		x_t.requires_grad_();
		torch::Tensor pred_energy=schnet.forward(ztype_t,x_t,edge_index1_t,edge_index2_t,CUTOFF1,CUTOFF2,GAMMA,RBF_KERNEL_NUM,HID_DIM,EXPONENT)*100;
		torch::Tensor Ft=-torch::autograd::grad({pred_energy[0]},{x_t},{torch::ones_like(pred_energy[0])*100},true)[0];
		double ptemp;
		ptemp = pred_energy[0].item().toDouble();
		if (eflag)
		{
			if(eflag_global)	eng_vdwl += ptemp;
			if(eflag_atom)		eatom[i]  = ptemp;
		}
		f[i][0] += Ft[0][0].item().toDouble();
		f[i][1] += Ft[0][1].item().toDouble();
		f[i][2] += Ft[0][2].item().toDouble();
		for (jj = 0; jj < jnum; jj++)
		{
			j = jlist[jj];
			j &= NEIGHMASK;
			for(int hcx=0;hcx<edge_num[i];hcx++)
			{
				if(j==graphmap[hcx])
				{
					f[j][0] += Ft[hcx+1][0].item().toDouble();
					f[j][1] += Ft[hcx+1][1].item().toDouble();
					f[j][2] += Ft[hcx+1][2].item().toDouble();
				}
			}
		}
		memory->destroy(graphmap);
	}
	memory->destroy(edge_num);
	//comm->forward_comm_pair(this);
	if (vflag_fdotr) virial_fdotr_compute();
}
 /* ---------------------------------------------------------------------- */

void PairGCNFF1::allocate()
{
  //printf("allocate: start"); 
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  
  map = new int[n+1];
  //printf("allocate: end"); 
}

/* ----------------------------------------------------------------------
   global settings
   ------------------------------------------------------------------------- */

void PairGCNFF1::settings(int narg, char **arg)
{
 // printf("setting: start"); 
  if (narg != 0) error->all(FLERR,"Illegal pair_style command");
 // printf("setting: end"); 
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
   ------------------------------------------------------------------------- */

void PairGCNFF1::coeff(int narg, char **arg)
{
  //printf("coeff: start"); 
  int i,j,n;

  if (!allocated) allocate();

  if (narg != 3 + atom->ntypes)
    error->all(FLERR,"Incorrect args for pair coefficients");

  // insure I,J args are * *

  if (strcmp(arg[0],"*") != 0 || strcmp(arg[1],"*") != 0)
    error->all(FLERR,"Incorrect args for pair coefficients");
   
  // XQ: read elements and nelements from lammps input  
  // read args that map atom types to elements in potential file
  // map[i] = which element the Ith atom type is, -1 if NULL
  // nelements = # of unique elements
  // elements = list of element names

  if (elements) {
    for (i = 0; i < nelements; i++) delete [] elements[i];
    delete [] elements;
  }
  elements = new char*[atom->ntypes];
  for (i = 0; i < atom->ntypes; i++) elements[i] = NULL;

  nelements = 0;
  for (i = 3; i < narg; i++) {
    if (strcmp(arg[i],"NULL") == 0) {
      map[i-2] = -1;
      continue;
    }
    for (j = 0; j < nelements; j++)
      if (strcmp(arg[i],elements[j]) == 0) break;
    map[i-2] = j;
    if (j == nelements) {
      n = strlen(arg[i]) + 1;
      elements[j] = new char[n];
      strcpy(elements[j],arg[i]);
      nelements++;
    }
  }

  // read potential file and initialize potential parameters
  read_file(arg[2]);
  setup_params();
	 Embedding_weight										= Array2Tensor(embedding_weight,nparams,HID_DIM);
	 Interaction_blocks_0_linear1_weight					= Array2Tensor(interaction_blocks_0_linear1_weight,HID_DIM,HID_DIM);
	 Interaction_blocks_0_linear1_bias						= Array2Tensor(interaction_blocks_0_linear1_bias,HID_DIM);
	 Interaction_blocks_0_filter_block_linear1_weight		= Array2Tensor(interaction_blocks_0_filter_block_linear1_weight,HID_DIM,RBF_KERNEL_NUM);
	 Interaction_blocks_0_filter_block_linear1_bias			= Array2Tensor(interaction_blocks_0_filter_block_linear1_bias,HID_DIM);
	 Interaction_blocks_0_filter_block_linear2_weight		= Array2Tensor(interaction_blocks_0_filter_block_linear2_weight,HID_DIM,HID_DIM);
	 Interaction_blocks_0_filter_block_linear2_bias			= Array2Tensor(interaction_blocks_0_filter_block_linear2_bias,HID_DIM);
	 Interaction_blocks_0_linear2_weight					= Array2Tensor(interaction_blocks_0_linear2_weight,HID_DIM,HID_DIM);
	 Interaction_blocks_0_linear2_bias						= Array2Tensor(interaction_blocks_0_linear2_bias,HID_DIM);
	 Interaction_blocks_0_linear3_weight					= Array2Tensor(interaction_blocks_0_linear3_weight,HID_DIM,HID_DIM);
	 Interaction_blocks_0_linear3_bias						= Array2Tensor(interaction_blocks_0_linear3_bias,HID_DIM);
	 Atomwise1_weight										= Array2Tensor(atomwise1_weight,HALF_HID_DIM,HID_DIM);
	 Atomwise1_bias											= Array2Tensor(atomwise1_bias,HALF_HID_DIM);
	 Atomwise2_weight										= Array2Tensor(atomwise2_weight,1,HALF_HID_DIM);
	 Atomwise2_bias											= Array2Tensor(atomwise2_bias,1);
	
  // clear setflag since coeff() called once with I,J = * *

  n = atom->ntypes;
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  // set setflag i,j for type pairs where both are mapped to elements

  int count = 0;
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      if (map[i] >= 0 && map[j] >= 0) {
        setflag[i][j] = 1;
        count++;
      }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
  //printf("coeff: end\n"); 
}

/* ---------------------------------------------------------------------- */

void PairGCNFF1::read_file(char *file)
{
	memory->sfree(params);
	params = NULL;
	nparams = 0;
	// open file on proc 0 only
	// then read line by line and broadcast the line to all MPI ranks
	FILE *fp;
	if (comm->me == 0)
	{
		fp = utils::open_potential(file,lmp,nullptr);
		if (fp == nullptr)
		{
			char str[128];
			sprintf(str,"Cannot open GCNFF1 potential file %s",file);
			error->one(FLERR,str);
		}
	}

	int i,j,n,nwords,curparam,wantdata;
	char line[MAXLINE],*ptr;
	int eof = 0;
	char **words = new char*[MAXWORD+1];

	while (1)
	{
		n = 0;
		if (comm->me == 0)
		{
			ptr = fgets(line,MAXLINE,fp);
			if (ptr == NULL)
			{
				eof = 1;
				fclose(fp);
			}
			else
				n = strlen(line) + 1;
		}

		MPI_Bcast(&eof,1,MPI_INT,0,world);
		if (eof) break;
		MPI_Bcast(&n,1,MPI_INT,0,world);
		MPI_Bcast(line,n,MPI_CHAR,0,world);
		// strip comment, skip line if blank

		if ((ptr = strchr(line,'#')))
			*ptr = '\0';
		nwords = utils::count_words(line);
		if (nwords == 0) continue;

		if (nwords > MAXWORD)
			error->all(FLERR,"Increase MAXWORD and recompile");

		// words = ptrs to all words in line

		nwords = 0;
		words[nwords++] = strtok(line," \t\n\r\f");
		while ((words[nwords++] = strtok(NULL," \t\n\r\f"))) continue;
		--nwords;
		/*
		if ((nwords == 2) && (strcmp(words[0],"generation") == 0))
		{
			int ver = atoi(words[1]);
			if (ver != GCNFF1_VERSION)
				error->all(FLERR,"Incompatible GCNFF1 potential file version");
			if ((ver == 1) && (nelements != 1))
				error->all(FLERR,"Cannot handle multi-element systems with this potential");
		}
		else */
		if ((nwords == 2) && (strcmp(words[0],"n_elements") == 0))
		{
			nparams = atoi(words[1]);														// nparams is the number of atom type  YY 
			//MPI_Bcast(&nparams,1,MPI_INT,0,world);
			if ((nparams < 1) || params)													// sanity check
				error->all(FLERR,"Invalid GCNFF1 potential file");
			params = memory->create(params,nparams,"pair:params");
			memset(params,0,nparams*sizeof(Param));
			curparam = -1;
		}
		else if ((nwords == 2) && (strcmp(words[0],"HID_DIM") == 0))
		{
			HID_DIM = atoi(words[1]);
			//MPI_Bcast(&HID_DIM,1,MPI_INT,0,world);
		}
		else if ((nwords == 2) && (strcmp(words[0],"CUTOFF1") == 0))
		{
			CUTOFF1 = atof(words[1]);
			//MPI_Bcast(&CUTOFF,1,MPI_DOUBLE,0,world);
		}
		else if ((nwords == 2) && (strcmp(words[0],"CUTOFF2") == 0))
		{
			CUTOFF2 = atof(words[1]);
			//MPI_Bcast(&CUTOFF,1,MPI_DOUBLE,0,world);
		}
		else if ((nwords == 2) && (strcmp(words[0],"EXPONENT") == 0))
		{
			EXPONENT = atof(words[1]);
			//MPI_Bcast(&RBF_KERNEL_NUM,1,MPI_INT,0,world);
		}
		else if ((nwords == 2) && (strcmp(words[0],"RBF_KERNEL_NUM") == 0))
		{
			RBF_KERNEL_NUM = atoi(words[1]);
			//MPI_Bcast(&RBF_KERNEL_NUM,1,MPI_INT,0,world);
		}
		else if ((nwords == 2) && (strcmp(words[0],"NUM_CONV") == 0))
		{
			NUM_CONV = atoi(words[1]);
			//MPI_Bcast(&NUM_CONV,1,MPI_INT,0,world);
		}
		else if ((nwords == 2) && (strcmp(words[0],"GAMMA") == 0))
		{
			GAMMA = atof(words[1]);
			//MPI_Bcast(&GAMMA,1,MPI_DOUBLE,0,world);
		}
		else if ((nwords == 1) && (strcmp(words[0],"Finish") == 0))
		{
			memory->create(embedding_weight,nparams,HID_DIM,"pair:embedding_weight");
			memory->create(interaction_blocks_0_linear1_weight,HID_DIM,HID_DIM,						"pair:interaction_blocks_0_linear1_weight");
			memory->create(interaction_blocks_0_linear1_bias,HID_DIM,								"pair:interaction_blocks_0_linear1_bias");
			memory->create(interaction_blocks_0_filter_block_linear1_weight,HID_DIM,RBF_KERNEL_NUM,	"pair:interaction_blocks_0_filter_block_linear1_weight");
			memory->create(interaction_blocks_0_filter_block_linear1_bias,HID_DIM,					"pair:interaction_blocks_0_filter_block_linear1_bias");
			memory->create(interaction_blocks_0_filter_block_linear2_weight,HID_DIM,HID_DIM,		"pair:interaction_blocks_0_filter_block_linear2_weight");
			memory->create(interaction_blocks_0_filter_block_linear2_bias,HID_DIM,					"pair:interaction_blocks_0_filter_block_linear2_bias");
			memory->create(interaction_blocks_0_linear2_weight,HID_DIM,HID_DIM,						"pair:interaction_blocks_0_linear2_weight");
			memory->create(interaction_blocks_0_linear2_bias, HID_DIM,								"pair:interaction_blocks_0_linear2_bias");
			memory->create(interaction_blocks_0_linear3_weight,HID_DIM,HID_DIM,						"pair:interaction_blocks_0_linear3_weight");
			memory->create(interaction_blocks_0_linear3_bias, HID_DIM,								"pair:interaction_blocks_0_linear3_bias");
			HALF_HID_DIM=int(HID_DIM/2);
			//MPI_Bcast(&HALF_HID_DIM,1,MPI_INT,0,world);
			memory->create(atomwise1_weight,HALF_HID_DIM,HID_DIM,									"pair:atomwise1_weight");
			memory->create(atomwise1_bias,HALF_HID_DIM,												"pair:atomwise1_bias");
			memory->create(atomwise2_weight,1,HALF_HID_DIM,											"pair:atomwise2_weight");
			memory->create(atomwise2_bias,1,														"pair:atomwise2_bias");
		}
		else if ((nwords>1)&&(strcmp(words[0],"embedding.weight") == 0))
		{
			for(i=0;i<nparams;i++)
				for(j=0;j<HID_DIM;j++)
				{
					int myorder=i*HID_DIM+j+1;
					embedding_weight[i][j]=atof(words[myorder]);
				}
			//MPI_Bcast(embedding_weight,nparams*HID_DIM, MPI_DOUBLE, 0, world);
		}
		
		else if ((nwords>1)&&(strcmp(words[0],"interaction_blocks.0.linear1.weight") == 0))
		{
			for(i=0;i<HID_DIM;i++)
				for(j=0;j<HID_DIM;j++)
				{
					int myorder=i*HID_DIM+j+1;
					interaction_blocks_0_linear1_weight[i][j]=atof(words[myorder]);
				}
			//MPI_Bcast(interaction_blocks_0_linear1_weight,HID_DIM*HID_DIM, MPI_DOUBLE, 0, world);
		}
		else if ((nwords>1)&&(strcmp(words[0],"interaction_blocks.0.linear1.bias") == 0))
		{
			for(j=0;j<HID_DIM;j++)
			{
				int myorder=j+1;
				interaction_blocks_0_linear1_bias[j]=atof(words[myorder]);
			}
			//MPI_Bcast(interaction_blocks_0_linear1_bias,HID_DIM, MPI_DOUBLE, 0, world); 
		}
		else if ((nwords>1)&&(strcmp(words[0],"interaction_blocks.0.cfconvlayer.filter_block.linear1.weight") == 0))
		{
			for(i=0;i<HID_DIM;i++)
				for(j=0;j<RBF_KERNEL_NUM;j++)
				{
					int myorder=i*RBF_KERNEL_NUM+j+1;
					interaction_blocks_0_filter_block_linear1_weight[i][j]=atof(words[myorder]);
				}
			//MPI_Bcast(interaction_blocks_0_filter_block_linear1_weight,HID_DIM*RBF_KERNEL_NUM, MPI_DOUBLE, 0, world);
		}
		else if ((nwords>1)&&(strcmp(words[0],"interaction_blocks.0.cfconvlayer.filter_block.linear1.bias") == 0))
		{
			for(j=0;j<HID_DIM;j++)
			{
				int myorder=j+1;
				interaction_blocks_0_filter_block_linear1_bias[j]=atof(words[myorder]);
			}
			//MPI_Bcast(interaction_blocks_0_filter_block_linear1_bias,HID_DIM, MPI_DOUBLE, 0, world);
		}
		else if ((nwords>1)&&(strcmp(words[0],"interaction_blocks.0.cfconvlayer.filter_block.linear2.weight") == 0))
		{
			for(i=0;i<HID_DIM;i++)
				for(j=0;j<HID_DIM;j++)
				{
					int myorder=i*HID_DIM+j+1;
					interaction_blocks_0_filter_block_linear2_weight[i][j]=atof(words[myorder]);
				}
			//MPI_Bcast(interaction_blocks_0_filter_block_linear2_weight,HID_DIM*HID_DIM, MPI_DOUBLE, 0, world);
		}
		else if ((nwords>1)&&(strcmp(words[0],"interaction_blocks.0.cfconvlayer.filter_block.linear2.bias") == 0))
		{
			for(j=0;j<HID_DIM;j++)
			{
				int myorder=j+1;
				interaction_blocks_0_filter_block_linear2_bias[j]=atof(words[myorder]);
			}
			//MPI_Bcast(interaction_blocks_0_filter_block_linear2_bias,HID_DIM, MPI_DOUBLE, 0, world);
		}
		else if ((nwords>1)&&(strcmp(words[0],"interaction_blocks.0.linear2.weight") == 0))
		{
			for(i=0;i<HID_DIM;i++)
				for(j=0;j<HID_DIM;j++)
				{
					int myorder=i*HID_DIM+j+1;
					interaction_blocks_0_linear2_weight[i][j]=atof(words[myorder]);
				}
			//MPI_Bcast(interaction_blocks_0_linear2_weight,HID_DIM*HID_DIM, MPI_DOUBLE, 0, world);
		}
		else if ((nwords>1)&&(strcmp(words[0],"interaction_blocks.0.linear2.bias") == 0))
		{
			for(j=0;j<HID_DIM;j++)
			{
				int myorder=j+1;
				interaction_blocks_0_linear2_bias[j]=atof(words[myorder]);
				}
			//MPI_Bcast(interaction_blocks_0_linear2_bias,HID_DIM, MPI_DOUBLE, 0, world);
		}
		else if ((nwords>1)&&(strcmp(words[0],"interaction_blocks.0.linear3.weight") == 0))
		{
			for(i=0;i<HID_DIM;i++)
				for(j=0;j<HID_DIM;j++)
				{
					int myorder=i*HID_DIM+j+1;
					interaction_blocks_0_linear3_weight[i][j]=atof(words[myorder]);
				}
			//MPI_Bcast(interaction_blocks_0_linear3_weight,HID_DIM*HID_DIM, MPI_DOUBLE, 0, world);
		}
		else if ((nwords>1)&&(strcmp(words[0],"interaction_blocks.0.linear3.bias") == 0))
		{
			for(j=0;j<HID_DIM;j++)
			{
				int myorder=j+1;
				interaction_blocks_0_linear3_bias[j]=atof(words[myorder]);
				}
			//MPI_Bcast(interaction_blocks_0_linear3_bias,HID_DIM, MPI_DOUBLE, 0, world); 
		}
		
		else if ((nwords>1)&&(strcmp(words[0],"atomwise1.weight") == 0))
		{
			for(i=0;i<HALF_HID_DIM;i++)
				for(j=0;j<HID_DIM;j++)
				{
					int myorder=i*HID_DIM+j+1;
					atomwise1_weight[i][j]=atof(words[myorder]);
				}
			//MPI_Bcast(atomwise1_weight,HALF_HID_DIM*HID_DIM, MPI_DOUBLE, 0, world);
		}
		else if ((nwords>1)&&(strcmp(words[0],"atomwise1.bias") == 0))
		{
			for(j=0;j<HALF_HID_DIM;j++)
			{
				int myorder=j+1;
				atomwise1_bias[j]=atof(words[myorder]);
			}
			//MPI_Bcast(atomwise1_bias,HALF_HID_DIM, MPI_DOUBLE, 0, world);
		}
		else if ((nwords>1)&&(strcmp(words[0],"atomwise2.weight") == 0))
		{
			for(i=0;i<1;i++)
				for(j=0;j<HALF_HID_DIM;j++)
				{
					int myorder=i*HALF_HID_DIM+j+1;
					atomwise2_weight[i][j]=atof(words[myorder]);
				}
			//MPI_Bcast(atomwise2_weight,1*HALF_HID_DIM, MPI_DOUBLE, 0, world);   
		}
		else if ((nwords>1)&&(strcmp(words[0],"atomwise2.bias") == 0))
		{
			for(j=0;j<1;j++)
			{
				int myorder=j+1;
				atomwise2_bias[j]=atof(words[myorder]);
			}
			//MPI_Bcast(atomwise2_bias,1*1, MPI_DOUBLE, 0, world); 
		}
		else if (params && (nwords == nparams+1) && (strcmp(words[0],"element") == 0))
		{
			wantdata = -1;
			for (i = 0; i < nparams; ++i)
			{
				for (j = 0; j < nelements; ++j)
					if (strcmp(words[i+1],elements[j]) == 0) break;
				if (j == nelements)
					error->all(FLERR,"No suitable parameters for requested element found");
				else params[i].ielement = j;
				//MPI_Bcast(&params[i].ielement, 1, MPI_INT, 0, world);
			}
		}
		else if (params && (nwords == 2) && (strcmp(words[0],"interaction") == 0))
		{
			for (i = 0; i < nparams; ++i)
			if (strcmp(words[1],elements[params[i].ielement]) == 0) curparam = i;
		}
		else if ((curparam >=0) && (nwords == 1) && (strcmp(words[0],"endVar") == 0))
		{
			wantdata = curparam;
			curparam = -1;
		}
		else if ((curparam >=0) && (nwords == 2) && (strcmp(words[0],"znum") == 0))
		{
			params[curparam].znum = atoi(words[1]);
			//MPI_Bcast(&params[curparam].znum, 1, MPI_INT, 0, world);
		}
		else if((curparam >=0) && (nwords == 2) && (strcmp(words[0],"Rc") == 0))
		{
			params[curparam].cut = atof(words[1]);
			//MPI_Bcast(&params[curparam].cut, 1, MPI_DOUBLE, 0, world);
		}
		else
		{
			if (comm->me == 0)
			error->warning(FLERR,"Ignoring unknown content in GCNFF1 potential file.");
		}
	}
	delete [] words;
	//printf("reading potential end\n");
}

/* ---------------------------------------------------------------------- */

void PairGCNFF1::setup_params()
{
  int i,m,n;
  double rtmp, rtmp2,rtmp3; // YY : 2 is for 3-body  3 is for m-body

  // set elem2param for all elements

  memory->destroy(elem2param);
  memory->create(elem2param,nelements,"pair:elem2param");

  for (i = 0; i < nelements; i++) {
    n = -1;
    for (m = 0; m < nparams; m++) {
      if (i == params[m].ielement) {
        if (n >= 0) error->all(FLERR,"Potential file has duplicate entry");
        n = m;
      }
    }
    if (n < 0) error->all(FLERR,"Potential file is missing an entry");
    elem2param[i] = n;
  }

  // compute parameter values derived from inputs

  // set cutsq using shortcut to reduce neighbor list for accelerated
  // calculations. cut must remain unchanged as it is a potential parameter
  // (cut = a*sigma)

  cutmax = 0.0;
  for (m = 0; m < nparams; m++) {
    rtmp = params[m].cut;
    if (rtmp > cutmax) cutmax = rtmp;
  }
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */
void PairGCNFF1::init_style()
{
	if(force->newton_pair == 0)
		error->all(FLERR,"Pair style meam/spline requires newton pair on");

		// Need both full and half neighbor list.
		int irequest_full = neighbor->request(this);
		neighbor->requests[irequest_full]->id = 1;
		neighbor->requests[irequest_full]->half = 0;
		neighbor->requests[irequest_full]->full = 1;
		int irequest_half = neighbor->request(this);
		neighbor->requests[irequest_half]->id = 2;
		neighbor->requests[irequest_half]->half = 0;
		//neighbor->requests[irequest_half]->half_from_full = 1;
		//neighbor->requests[irequest_half]->otherlist = irequest_full;
}

/* ----------------------------------------------------------------------
   neighbor callback to inform pair style of neighbor list to use
   half or full
------------------------------------------------------------------------- */
void PairGCNFF1::init_list(int id, NeighList *ptr)
{
        if(id == 1) listfull = ptr;
        else if(id == 2) listhalf = ptr;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */
double PairGCNFF1::init_one(int i, int j)
{
        //return cutoff;
		return cutmax;
}
/* ----------------------------------------------------------------------
        Transform the array to Tensor from
------------------------------------------------------------------------- */
torch::Tensor PairGCNFF1::Array2Tensor(int *array,int Larray)
{
	if(Larray==0) return torch::tensor({});
	else
	{
		torch::Tensor out_tensor = torch::tensor({array[0]});
		for(int i=1;i<Larray;i++)
			out_tensor=torch::cat({out_tensor,torch::tensor({array[i]})});
		return out_tensor;
	}
}

torch::Tensor PairGCNFF1::Array2Tensor(std::vector<int> array,int Larray)
{
	if(Larray==0) return torch::tensor({});
	else
	{
		torch::Tensor out_tensor = torch::tensor({array[0]});
		for(int i=1;i<Larray;i++)
			out_tensor=torch::cat({out_tensor,torch::tensor({array[i]})});
		return out_tensor;
	}
}

torch::Tensor PairGCNFF1::Array2Tensor(double *array,int Larray)
{
	torch::Tensor out_tensor = torch::tensor({array[0]});
	for(int i=1;i<Larray;i++)
		out_tensor=torch::cat({out_tensor,torch::tensor({array[i]})});
	return out_tensor;
}

torch::Tensor PairGCNFF1::Array2Tensor(double **array,int row,int cow)
{
	torch::Tensor out_tensor = torch::tensor({array[0][0]});
	for(int i=0;i<row;i++)
		for(int j=0;j<cow;j++)
		{
			if(i==0&&j==0)
				continue;
			else
				out_tensor=torch::cat({out_tensor,torch::tensor({array[i][j]})});
		}
	out_tensor=out_tensor.view({row,cow});
	return out_tensor;
}

torch::Tensor PairGCNFF1::Array2Tensor(int **array,int row,int cow)
{
	torch::Tensor out_tensor = torch::tensor({array[0][0]});
	for(int i=0;i<row;i++)
		for(int j=0;j<cow;j++)
		{
			if(i==0&&j==0)
				continue;
			else
				out_tensor=torch::cat({out_tensor,torch::tensor({array[i][j]})});
		}
	out_tensor=out_tensor.view({row,cow});
	return out_tensor;
}

/* ---------------------------------------------------------------------- */
void PairGCNFF1::grab(FILE *fptr, int n, double *list)
{
  char *ptr;
  char line[MAXLINE];

  int i = 0;
  while (i < n) {
    fgets(line,MAXLINE,fptr);
    ptr = strtok(line," \t\n\r\f");
    list[i++] = atof(ptr);
    while (ptr = strtok(NULL," \t\n\r\f")) list[i++] = atof(ptr);
  }
}

int PairGCNFF1::pack_forward_comm(int n, int *list, double *buf,
                                      int pbc_flag, int *pbc)
{
  int i,j,k,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
        for (k = 0; k < 8; k++)
          buf[m++] = rho_v[j][k];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void PairGCNFF1::unpack_forward_comm(int n, int first, double *buf)
{
    int i,k,m,last;

  m = 0;
 last = first + n;
  for (i = first; i < last; i++){
   for (k = 0; k < 8; k++)
      rho_v[i][k] = buf[m++];
    }


}

/* ---------------------------------------------------------------------- */

int PairGCNFF1::pack_reverse_comm(int n, int first, double *buf)
{
  int i,k,m,last;

   m = 0;
  last = first + n;
  for (i = first; i < last; i++)
         for (k = 0; k < 8; k++)
          buf[m++] = rho_v[i][k];
  return m;
}

/* ---------------------------------------------------------------------- */

void PairGCNFF1::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,k,m;

  m = 0;
  for (i = 0; i < n; i++) {
      j = list[i];
   for (k = 0; k < 8; k++)
      rho_v[j][k] = buf[m++];
    }
}

/* ----------------------------------------------------------------------
   Returns memory usage of local atom-based arrays
------------------------------------------------------------------------- */
double PairGCNFF1::memory_usage()
{
        return nmax *8* sizeof(double);
}


