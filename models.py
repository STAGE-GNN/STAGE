from collections import defaultdict
from pickle import FALSE
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.activation import PReLU
from torch_geometric.nn import GCNConv, GATConv,RGCNConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, uniform
from torch_geometric.utils import softmax
import math

from torch_sparse.tensor import to
from utils import *
from torch_sparse import SparseTensor, set_diag
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree, to_undirected
from torch.nn import Sequential, Linear, ReLU, Dropout
# from torch_scatter import scatter
import numpy as np
import dgl


class HyperGNN(nn.Module):
    def __init__(self,input_dim,output_dim,hyper_edge_num=1,num_layer=1,negative_slope=0.2):
        super(HyperGNN,self).__init__()
        self.negative_slope=negative_slope
        
        self.proj=nn.Linear(input_dim,output_dim,bias=False)
       
        self.alpha=nn.Parameter(torch.ones(hyper_edge_num,1))
        
        glorot(self.alpha)

    def forward(self,company_emb,hyp_graph):
        outlist=[]
        for i in range(len(hyp_graph)):
            laplacian=scipy_sparse_mat_to_torch_sparse_tensor(hyp_graph[i].laplacian()).to('cuda:0')
            rs= laplacian@self.proj(company_emb)
            outlist+=[rs]
           
        res=0
       
        alpha=torch.sigmoid(self.alpha)
       
        for i in range(len(outlist)):
            res+=outlist[i]*alpha[i]
        return res
    

import dgl
import dgl.nn as dglnn

# RGCN
class HeteGNN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names,rel_num):
        super().__init__()
        self.rel_num = rel_num
        self.proj_com=nn.Linear(in_feats,in_feats,bias=False)
        self.conv1 = dglnn.HeteroGraphConv({rel:dglnn.GraphConv(in_feats, hid_feats)
                                            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({rel:dglnn.GraphConv(hid_feats, out_feats)
                                           for rel in rel_names}, aggregate='sum')
        self.bn = nn.BatchNorm1d(in_feats)
    def forward(self, graph, inputs): 
        # inputs:  dict {'comp': comp_feat}
        # graph.nodes['company'].data['feature'] = self.proj_com(graph.nodes['company'].data['feature'])
        # graph.nodes['company'].data['feature'] = self.bn(graph.nodes['company'].data['feature'])
        h = self.conv1(graph, inputs)
        h = {k:F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h
    
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers)
    def forward(self, x):
        x = self.gru(x)
        x_m = torch.mean(x[0], dim=1) #5317*5*8
        print(1111111111111)
        return x_m
    
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional, output_size):
        super(RNNModel,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)
    def forward(self, x):
        output, hn = self.rnn(x)
        h = hn[-(1 + int(self.bidirectional)):]
        x = torch.cat(h.split(1), dim=-1).squeeze(0)
        x = self.fc(x)
        return x

    

class HyperSTGNN(nn.Module):
    def __init__(self,input_dim,output_dim,
     company_num,rel_num,
     device,com_initial_emb,g,node_features, #  node_features
     fin_seq,
     num_heads=1,dropout=0.2,norm=True,
     ):
        super(HyperSTGNN,self).__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.company_num=company_num
        # self.person_num=person_num
        self.rel_num=rel_num
        self.device=device

        self.num_heads=num_heads
        self.dropout=dropout
        self.norm=norm
        self.company_emb=com_initial_emb.to(device)
        # self.company_emb=torch.FloatTensor(com_initial_emb.float()).to(device) #

        # assert not np.any(np.isnan(com_initial_emb.detach().numpy())),'nan exists!'
        # self.person_emb=torch.FloatTensor(person_initial_emb)

        #risk data: dict-->{company_index:[[cause type, court type, result category, time(months),time_label],...] }        
        # self.riskinfo=RiskInfo(input_dim,company_num,cause_type_num,court_type_num,category_num,time_label_num=time_label_num)
        self.graph = g
        self.node_features = node_features

        self.fin_seq = fin_seq

        self.hypergnn= HyperGNN(input_dim,output_dim,num_layer=1)
        self.hetegnn = HeteGNN(in_feats=input_dim, hid_feats=2*input_dim, 
             out_feats=output_dim, rel_names=g.etypes,rel_num=rel_num) # out_feats
        
        self.rnn = RNNModel(input_size=24,hidden_size=64,num_layers=2,bidirectional=True,output_size=8)

        self.company_proj=nn.Linear(input_dim,input_dim,bias=False)
        # self.person_proj=nn.Linear(32,input_dim,bias=False)

        self.risk_proj=nn.Linear(input_dim,input_dim,bias=False)
        self.info_proj=nn.Linear(output_dim,output_dim,bias=False)

        self.fusion = nn.Linear(output_dim*2, output_dim, bias= True)

        self.final_proj=nn.Sequential(nn.Linear(input_dim,output_dim,bias=False),nn.ReLU(),nn.Linear(output_dim,output_dim,bias=False))
        self.alpha=torch.ones((1))
        # self.alpha=torch.zeros((1)) 

    # risk data: dict-->{company_index:[[cause type, court type, category, time(months),time_label],...] }
    # company attribute information: np.array()-->[[register_captial, paid_captial, set up time(months)]]
    # graph: edge index:[sour,tar].T -->2xN; edge type: [,,...,] -->N; edge weight:[,,...,]-->N
    # hyper graph: dict:{industry:{ind1:[...],ind2:[...],...},area:{area1:[...],area2:[...],...},qualify:{qua1:[...],qua2:[...],...}}
    def forward(self,hete_graph,node_features,hyp_graph,idx): #  node_features

        # assert not np.any(np.isnan(self.company_emb.detach().numpy())),'nan exists!'

        company_emb=self.company_proj(self.company_emb.float()) 
        # person_emb=self.person_proj(self.person_emb)
        # company_basic_info=torch.zeros((self.company_num,len(company_attr[0])))
       
        # company_basic_info[idx]=torch.Tensor(company_attr)
        
        # company_emb=torch.cat((company_emb,company_basic_info),dim=1)
        # risk_info=self.riskinfo(risk_data)
        # company_emb_info=self.risk_proj(torch.cat((company_emb,risk_info),dim=1)) 

        fin_seq_info = self.rnn(self.fin_seq)

        company_emb_info = company_emb

        # assert not np.any(np.isnan(company_emb_info.detach().numpy())),'nan exists!'

        company_emb_hyper=self.hypergnn(company_emb_info,hyp_graph)

        # edge_index,edge_type,edge_weight=hete_graph

        # g, node_features = hete_graph  #  node_features

        company_emb_hete = self.hetegnn(hete_graph,node_features) #


        # assert not np.any(np.isnan(company_emb_hyper.detach().numpy())),'nan exists!'

        company_emb_final=self.info_proj(company_emb_hyper+company_emb_hete['company']) ## batch,8
        # company_emb_final=self.info_proj(company_emb_hyper) #  hete
        # company_emb_final=self.info_proj(company_emb_hete['company']) #  hyper

        # company_emb_and_seq_info = self.final_proj(company_emb_info) # history seq
        company_emb_and_seq_info = self.fusion(torch.cat((self.final_proj(company_emb_info),fin_seq_info),dim=1))


        # assert not np.any(np.isnan(company_emb_final.detach().numpy())),'nan exists!'

        alpha=torch.sigmoid(self.alpha).to('cuda:0') # 0.7/0.5

        # company_emb_final=(alpha)*F.gelu(company_emb_final)+(1-alpha)*company_emb_and_seq_info # 
        company_emb_final=(1-alpha)*F.gelu(company_emb_final)+(alpha)*company_emb_and_seq_info 

        return company_emb_final[idx]