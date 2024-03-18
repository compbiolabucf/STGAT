import numpy as np
import pandas as pd
import torch
import glob
import os
import pandas as pd
import numpy as np
import torch

from .models import *
from .utils import *


seed = 0
np.random.seed(seed)
torch.manual_seed(seed)


# ------------------- Test TCGA --------------------- #

def label_tcga(model, tcga_images, tcga_name, spot_names, save_dir, threshold=0.5, gpu='cuda:0', BATCH_SIZE=16):

    first = True
    model.eval()

    torch.cuda.empty_cache()
    loader = tcga_loader(tcga_images, BATCH_SIZE, gpu)
    
    for batch in loader:
        output = model(batch)
        batch = batch.detach().cpu()
        output = output.detach().cpu()
        
        if first: 
            outputs = output.numpy()
            first=False
        else: outputs = np.append(outputs, output.numpy(), axis=0)                
        torch.cuda.empty_cache()

    prediction = np.array([1 if p>=threshold else 0 for p in outputs[:,1]])

    labels = pd.Series(prediction, index = spot_names, name='TCGA_prediction')
    labels.to_csv(save_dir + tcga_name + '_labels.csv')
    
    torch.cuda.empty_cache()
    print('----------------------------------------------------------------------------------------')




def apply_SLP(root_dir, BATCH_SIZE=16, gpu='cuda:0'):

    model_name = glob.glob('./trained/slp_trained_thresh_*.pkl')
    # defining the model
    model = classifier(in_channels=1)
    criterion = F1_Loss()

    # checking if GPU is available
    if torch.cuda.is_available():
        model = model.cuda(gpu)
        criterion = criterion.cuda(gpu)

    model.load_state_dict(torch.load(model_name[0], map_location='cpu')) 

    threshold = float(model_name[0].split('/')[2].split('.')[0].split('_')[-1])

    # root_dir = '../io_data/TCGA/'
    patches_dir = root_dir + '/patches/'
    save_dir = root_dir + 'clinical/'
    os.makedirs(save_dir, exist_ok=True)


    for name in os.listdir(patches_dir):  
        print('------Generating labels for sample: {}---------'.format(name))
        spot_names, tcga_data = load_tcga(patches_dir+name+'/')
        label_tcga(model, tcga_data, name, spot_names, save_dir, threshold=threshold, gpu=gpu, BATCH_SIZE=BATCH_SIZE)
