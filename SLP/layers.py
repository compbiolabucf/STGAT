import torch.nn as nn
import torch.nn.functional as F



def conv_block(in_channels, out_channels, pool_size=2, k=3, p=1, pool=False):
    conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=p, stride=1)
    nn.init.xavier_uniform_(conv_layer.weight, gain=1)
    nn.init.constant_(conv_layer.bias, val=0)
    layers = [conv_layer, 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(kernel_size = pool_size, stride = 2))
    return nn.Sequential(*layers)




def linear_block(in_features, out_features, final_layer=False):
    linear_layer = nn.Linear(in_features, out_features, bias=True)
    nn.init.xavier_uniform_(linear_layer.weight, gain=1)
    nn.init.constant_(linear_layer.bias, val=0)

    if final_layer:
        return linear_layer
    
    layers = [linear_layer, 
              nn.BatchNorm1d(out_features), 
              nn.ReLU(inplace=True)]

    return nn.Sequential(*layers)