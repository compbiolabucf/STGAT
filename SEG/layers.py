import torch
import torch.nn as nn
import torch.nn.functional as F



def conv_block(in_channels, out_channels, relu=True, pool_size=2, pool=False):
    conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
    nn.init.xavier_uniform_(conv_layer.weight, gain=1)
    nn.init.constant_(conv_layer.bias, val=0)

    if relu:
        layers = [conv_layer, 
              nn.ReLU(inplace=True)]
    else:
        layers = [conv_layer]
        
    if pool: layers.append(nn.MaxPool2d(kernel_size = pool_size, stride = 2))
    
    return nn.Sequential(*layers)




def linear_block(in_features, out_features, final_layer=False):
    linear_layer = nn.Linear(in_features, out_features, bias=True)
    nn.init.xavier_uniform_(linear_layer.weight, gain=1)
    nn.init.constant_(linear_layer.bias, val=0)

    if final_layer:
        return linear_layer
    else:
        layers = [linear_layer, 
                nn.ReLU(inplace=True)]

    return nn.Sequential(*layers)





class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat 

        self.W_s = nn.Parameter(torch.empty(size=(in_features, out_features)))   ## declaring the weights for linear transformation
        nn.init.xavier_uniform_(self.W_s.data, gain=0.5)                           ## initializing the linear transformation weights from the uniform distribution U(-a,a)
        self.a_s = nn.Parameter(torch.empty(size=(out_features, 1)))           ## declaring weights for creating self attention coefficients
        nn.init.xavier_uniform_(self.a_s.data, gain=0.5)                           ## initializing the attention-coefficient weights
        
        self.W_n = nn.Parameter(torch.empty(size=(in_features, out_features)))   ## declaring the weights for linear transformation
        nn.init.xavier_uniform_(self.W_n.data, gain=0.5)                           ## initializing the linear transformation weights from the uniform distribution U(-a,a)
        self.a_n = nn.Parameter(torch.empty(size=(out_features, 1)))           ## declaring weights for creating self attention coefficients
        nn.init.xavier_uniform_(self.a_n.data, gain=0.5)

        self.relu = nn.ReLU(inplace=True)  ## changed LeakyReLU

    def forward(self, input, adj):
        h_s = torch.mm(input, self.W_s)    ## multiplying inputs with the weights for linear transformation with dimension (#input X out_features)
        h_n = torch.mm(input, self.W_n)    ## multiplying inputs with the weights for linear transformation with dimension (#input X out_features)        
        
        e = self._prepare_attentional_mechanism_input(h_s, h_n)

        zero_vec = -9e15*torch.ones_like(e)   #torch.zeros_like(e)                       
        attention = torch.where(adj > 0, e, zero_vec)             ## assigning values of 'e' to those which has value>0 in adj matrix
        attention = F.softmax(attention, dim=1) 

        attention_sum = torch.sum(attention, axis = -1, keepdims = True)
        attention_s = 1/(1 + attention_sum)  
        attention_n = attention

        h_s = F.dropout(h_s, self.dropout, training=self.training)
        h_n = F.dropout(h_n, self.dropout, training=self.training)

        h_prime = attention_s*h_s + torch.matmul(attention_n, h_n)                      ## multiplying attention co-efficients with the input  -- dimension (#input X out_features)

        if self.concat:
            xtra = F.elu(h_prime)
            return F.elu(h_prime)
        else:
            return h_prime
            
    def _prepare_attentional_mechanism_input(self, h_s, h_n):
        
        h1 = torch.matmul(h_s, self.a_s)
        h2 = torch.matmul(h_n, self.a_n)
        e = h1 + h2.T   # broadcast add
        return self.relu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
