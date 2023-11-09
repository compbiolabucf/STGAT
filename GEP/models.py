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

    








class GEP(nn.Module):
    def __init__(self, nb_genes, alpha=0.2, gpu='cuda', nb_heads=16, nb_embed=64):
        super(GEP, self).__init__()
        # Using a pretrained model
        self.alpha = alpha
        self.spot_net = SEG(in_channels = 1, nb_genes = nb_genes, nb_heads = nb_heads, nb_embed = nb_embed)
        self.spot_net.load_state_dict(torch.load('./trained/{}.pkl'.format('sp_trained'), map_location=torch.device(gpu)))
        self.spot_out_features = self.spot_net.linear_layers[0][0].in_features
        # self.spot_process = linear_block(self.spot_out_features, nb_genes)

        self.spot_multiplier = nn.Parameter(nn.init.xavier_uniform_(
            torch.empty(self.spot_out_features, self.spot_out_features), 
            gain=1))
        self.tcga_multiplier = nn.Parameter(nn.init.xavier_uniform_(
            torch.empty(self.spot_out_features, self.spot_out_features), 
            gain=1))


        self.spot_net.linear_layers = nn.Sequential()
        
        self.tcga_preprocess = nn.Sequential(
                                    linear_block(nb_genes, int(self.spot_out_features/2)),
                                    linear_block(int(self.spot_out_features/2), self.spot_out_features, final_layer=True)
                                )

        self.encoder = nn.Sequential(
                            linear_block(self.spot_out_features, int(self.spot_out_features*2)),
                            linear_block(int(self.spot_out_features*2), int(nb_genes/2)),
                            linear_block(int(nb_genes/2), nb_genes, final_layer=True)
                            )

    
    def forward(self, x, adj, tcga_exp, cnn=True):
        if cnn:
            x = self.spot_net.cnn_layers(x)
            x = self.spot_net.intermediate_layer(x)
            return x

        else:
            x = self.spot_net.gat(x, adj)
            x = self.spot_net.linear_layers(x)

        # x = self.spot_process(x)
        tcga_exp = self.tcga_preprocess(tcga_exp)
        spot_processed = torch.matmul(x,self.spot_multiplier)
        tcga_processed = torch.matmul(tcga_exp,self.tcga_multiplier)
        
        spot_processed = (spot_processed-spot_processed.mean(dim=0))/spot_processed.std(dim=0)
        tcga_processed = (tcga_processed-tcga_processed.mean(dim=0))/tcga_processed.std(dim=0)
        x = spot_processed + tcga_processed
        x = self.encoder(x)

        return x

    
    def freeze(self):
        # To freeze the residual layers
        for param in self.spot_net.parameters():
            param.require_grad = False
    
    def unfreeze(self):
        # To unfreeze all layers
        for param in self.net.parameters():
            param.require_grad = True

