import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim

import time
import glob
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from .models import *
from .utils import *


seed = 0
np.random.seed(seed)
torch.manual_seed(seed)



def train_model(model, train_data, val_data, criterion, gpu='cuda:0', BATCH_SIZE=16, lr=1e-5, epochs=1000, patience=30, lr_step=10, lr_gamma=0.7, grad_clip=None):

    t_total = time.time()
    best = 1e100

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)   #1e-6
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    loss_vals = []
    os.makedirs('saved_model/',exist_ok=True)
    for epoch in range(epochs):        

        print('Current learning rate: ', scheduler.get_last_lr())
        train_losses = []
        first = True
        
        # clearing the Gradients of the model parameters
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        loader = get_dataloader(train_data, BATCH_SIZE, gpu)
        for batch in loader:
            
            output = model(batch[0])
            loss_train = criterion(output, batch[1])
            loss_train.backward()
            batch[0] = batch[0].detach().cpu()
            batch[1] = batch[1].detach().cpu()
            output = output.detach().cpu()
            
            if first: 
                outputs = output.numpy()
                labels = batch[1].numpy()
                first=False
            else: 
                outputs = np.append(outputs, output.numpy(), axis=0)
                labels = np.append(labels, batch[1].numpy(), axis=0)

            # Gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            torch.cuda.empty_cache()
            train_losses.append(loss_train.detach().cpu())

        tr_loss = np.mean(np.array(train_losses))
        tr_score = roc_score(outputs, labels)
        scheduler.step()

        print('Epoch : ',epoch, '\t', 'train loss:', tr_loss, '\t', 'train AUROC: ', tr_score)
        torch.cuda.empty_cache()

        val_l,_,_ = validate_model(model, val_data, criterion=criterion, gpu=gpu, BATCH_SIZE=BATCH_SIZE)
        loss_vals.append(val_l)
        torch.save(model.state_dict(), 'saved_model/{}.pth'.format(epoch))
        if loss_vals[-1] < best:
            best = loss_vals[-1]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == patience:
            break

        files = glob.glob('saved_model/*.pth')
        for file in files:
            epoch_nb = int(file.split('/')[1].split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(file)

        print('----------------------------------------------------------------------------------------')
        

    files = glob.glob('saved_model/*.pth')
    for file in files:
        epoch_nb = int(file.split('/')[1].split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(file)    

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Restore best model
    print('Loading {}th epoch'.format(best_epoch))
    model.load_state_dict(torch.load('saved_model/{}.pth'.format(best_epoch)))
    
    return model






def validate_model(model, val_data, criterion, gpu='cuda:0', BATCH_SIZE=16):
    val_losses = []
    first = True
    model.eval()

    torch.cuda.empty_cache()
    loader = get_dataloader(val_data, BATCH_SIZE, gpu)
    for _, batch in enumerate(loader):
        output = model(batch[0])
        val_losses.append(criterion(output,batch[1]).item())
        batch[0] = batch[0].detach().cpu()
        batch[1] = batch[1].detach().cpu()
        output = output.detach().cpu()

        if first: 
            outputs = output.numpy()
            labels = batch[1].numpy()
            first=False
        else: 
            outputs = np.append(outputs, output.numpy(), axis=0)
            labels = np.append(labels, batch[1].numpy(), axis=0)
                
        torch.cuda.empty_cache()

    val_loss = np.mean(np.array(val_losses))
    print("validation loss: ", val_loss, "\tvalidation AUROC: ", roc_score(outputs, labels))
    torch.cuda.empty_cache()

    return val_loss, outputs, labels
    



def test_model(model, test_data, criterion, pred_thresh = 0.5, gpu='cuda:0', BATCH_SIZE=16):
    print('-----------------------------Testing--------------------------------------')

    test_losses = []
    first = True
    model.eval()

    torch.cuda.empty_cache()
    loader = get_dataloader(test_data, BATCH_SIZE, gpu)
    for _, batch in enumerate(loader):
        output = model(batch[0])
        test_losses.append(criterion(output,batch[1]).item())
        batch[0] = batch[0].detach().cpu()
        batch[1] = batch[1].detach().cpu()
        output = output.detach().cpu().numpy()


        output = [0 if output[i][1]<pred_thresh else 1 for i in range(output.shape[0])]        
        
        if first: 
            outputs = output
            labels = batch[1].numpy()
            first=False
        else: 
            outputs = np.append(outputs, output, axis=0)
            labels = np.append(labels, batch[1].numpy(), axis=0)
                
        torch.cuda.empty_cache()
        

    print("Test Loss: ", np.mean(np.array(test_losses)))
    print("Test AUROC: ", roc_auc_score(labels, outputs))

    torch.cuda.empty_cache()
    print('----------------------------------------------------------------------------------------')





def check_youden(model, val_data, criterion, gpu='cuda:0', BATCH_SIZE=16):
    thresh_vals = np.linspace(0.3,0.8,20)
    _, outputs, labels = validate_model(model, val_data, criterion=criterion, gpu=gpu, BATCH_SIZE=BATCH_SIZE)
    y_ind = []

    print('\n\n.......Youden\'s index checking......\n')
    for threshold in thresh_vals:
        sensitivity, specificity = sense_spec(outputs, labels, threshold)
        y_ind.append((sensitivity+specificity-1))
    return thresh_vals[np.argmax(y_ind)]



def load_data(names, root_dir, patch_dir):

    label_dir = root_dir + '/clinical/'

    Sample_labels = []
    Sample_images = []
    
    for f in names:
        f_name = f.split('.')[0]
        print('\nLoading sample: ', f_name)
        img_dir = patch_dir + f_name + '/'
        
        labels = pd.read_csv(label_dir + f_name + '_clinical.csv', index_col=0)
        labels = labels.replace(to_replace={'tumor', 'non'}, value={1,0})
        Sample_labels.append(labels)
        Sample_images.append(load_spot_images(img_dir, labels.index))
        
    return process_separate(Sample_images, Sample_labels)




# train and test SLP module
def run_SLP(root_dir, patience, n_epochs, lr, lr_step, lr_gamma, BATCH_SIZE, gpu='cuda:0'):
    
    patch_dir = root_dir + '/patches/'
    names = os.listdir(patch_dir)
    tr_x,test_names=train_test_split(names, test_size=0.3,random_state=seed)
    train_names,val_names=train_test_split(tr_x,test_size=0.3,random_state=seed)

    train_data = load_data(train_names, root_dir, patch_dir)
    test_data = load_data(test_names, root_dir, patch_dir)
    val_data = load_data(val_names, root_dir, patch_dir)

    ## freeing up cache
    torch.cuda.empty_cache()

    # defining the model
    model = classifier(in_channels=1)
    criterion = F1_Loss()
    
    # checking if GPU is available
    if torch.cuda.is_available():
        model = model.cuda(gpu)
        criterion = criterion.cuda(gpu)

    # training
    model = train_model(model, train_data, val_data, criterion, gpu=gpu, BATCH_SIZE=BATCH_SIZE, lr=lr, lr_step=lr_step, lr_gamma=lr_gamma, epochs=n_epochs, patience=patience, grad_clip=None)    
    threshold = check_youden(model, val_data, criterion)
    os.makedirs('./trained/', exist_ok=True)
    
    for file in glob.glob('./trained/slp_trained_thresh_*.pkl'): os.remove(file)
    torch.save(model.state_dict(), './trained/{}.pkl'.format('slp_trained_thresh_{}'.format(threshold)))

    # testing
    test_model(model, test_data,  criterion, pred_thresh=threshold, gpu=gpu, BATCH_SIZE=BATCH_SIZE)