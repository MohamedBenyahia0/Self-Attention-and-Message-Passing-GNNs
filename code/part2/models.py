"""
Deep Learning on Graphs - ALTEGRAD - Nov 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, n_class, device):
        super(GNN, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_3)
        self.fc4 = nn.Linear(hidden_dim_3, n_class)
        self.relu = nn.ReLU()

    def forward(self, x_in, adj, idx):
        
        ############## Task 6
    
        ##################
        

        h=self.fc1(x_in)
    
        h=self.relu(torch.mm(adj,h))
        h=self.fc2(h)
        h=self.relu(torch.mm(adj,h))
 
        ##################
       
        
        idx = idx.unsqueeze(1).repeat(1, h.size(1))
        out = torch.zeros(idx.max()+1, h.size(1)).to(self.device)
        out = out.scatter_add_(0, idx, h) 
        
        ##################
        out=self.fc3(out)
        out=self.relu(out)
        out=self.fc4(out)  
        ##################
        
        

        return F.log_softmax(out, dim=1)
