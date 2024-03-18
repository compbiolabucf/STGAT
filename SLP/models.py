import torch.nn as nn
import torch.nn.functional as F
from .layers import * 




class classifier(nn.Module):
    def __init__(self, in_channels, dropout=0.2):
        super(classifier, self).__init__()
        self.dropour = dropout

        self.cnn_layers = nn.Sequential(
                            conv_block(in_channels, 8, k=3, p=1, pool=True),        #112      ## output 224*224
                            conv_block(8,32, k=3, p=1, pool=True),       #56       ## output: 112*112
                            conv_block(32,128, pool=True),     #28       ## output: 112*112
                            conv_block(128,512, pool=True),     #14       ## output: 56*56
                            conv_block(512, 1024, pool=True))   #7
        self.linear_layers = nn.Sequential(
                                nn.MaxPool2d(7),
                                nn.Flatten(),
                                linear_block(1024, 256),
                                linear_block(256, 32),
                                linear_block(32, 2, final_layer=True),
                                nn.Softmax(dim=1))
    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.linear_layers(x)
        return x