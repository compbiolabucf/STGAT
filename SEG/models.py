import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import * 




class GAT(nn.Module):
    def __init__(self, in_feat, embed_size, nheads=8, dropout=0.6, l_alpha=0.2):
        super(GAT, self).__init__()
        self.dropout = dropout

        ## creating attention layers for given number of heads
        self.attentions = [GraphAttentionLayer(in_feat, embed_size, dropout=dropout, alpha=l_alpha, concat=True) for _ in range(nheads)] 
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)     ## adding the modules for each head

    def forward(self, x, adj):
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)  

        return x  







class SEG(nn.Module):   
    def __init__(self, in_channels, nb_genes, nb_heads, nb_embed, dropout=0, l_alpha=0.2):
        super(SEG, self).__init__()
        
        # self.dropout = dropout
        self.cnn_layers = nn.Sequential(
              conv_block(in_channels,4, pool=False),       
              conv_block(4,4, pool=True),       
              conv_block(4,8, pool=True),    
              conv_block(8,8, pool=True),    
              conv_block(8,16, pool=True),    
              conv_block(16,16, pool=True)) 


        ## Preparing the GAT layer
        self.gat_in_features = 16 * 7 * 7
        self.intermediate_layer = nn.Sequential(
                                nn.Flatten(),
                                linear_block(in_features=self.gat_in_features, out_features=self.gat_in_features))
  
        ## Creating GAT layer
        self.gat = GAT(in_feat=self.gat_in_features, 
              embed_size=nb_embed, 
              nheads=nb_heads, 
              dropout=dropout, 
              l_alpha=l_alpha)


        ## Predicting gene expression from embedding
        self.linear_in_features = int(nb_embed*nb_heads)
        self.linear_layers = nn.Sequential(
            linear_block(self.linear_in_features, int(self.linear_in_features*2)),
            linear_block(int(self.linear_in_features*2), int(nb_genes/2)),
            linear_block(int(nb_genes/2), nb_genes, final_layer=True)
        )
        

    # Defining the forward pass    
    def forward(self, x, adj, cnn=True):
        if cnn:
            x = self.cnn_layers(x)
            x = self.intermediate_layer(x)

        else:
            x = self.gat(x, adj)
            x = self.linear_layers(x)

        return x
