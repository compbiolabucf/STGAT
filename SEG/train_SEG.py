import time
import glob
import os
import numpy as np
import pandas as pd

import torch
from torch.autograd import Variable
from torch import nn, optim

from sklearn.model_selection import train_test_split


from .models import *
from .utils import *


seed = 0
np.random.seed(seed)
torch.manual_seed(seed)



def single_sample_run(model, image_set, dataset, adj, test=False, BATCH_SIZE=16, gpu='cuda:0'):
    torch.cuda.empty_cache()
    loader = get_dataloader(image_set, BATCH_SIZE, gpu)

    cnn_set = Variable(torch.zeros(dataset.shape[0], 16*7*7).cuda(gpu))
    # print('image spots: ', cnn_set.shape[0])   ## remove
    
    for i, batch in enumerate(loader):
        cnn_out = model(batch, adj, cnn=True)
        if (i+1)*BATCH_SIZE>=image_set.shape[0]:y_batch = cnn_set[i*BATCH_SIZE:] = cnn_out
        else: cnn_set[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = cnn_out

        batch = batch.detach().cpu()
        cnn_out = cnn_out.detach().cpu()
        torch.cuda.empty_cache()

    print('After only CNN: ', cnn_set)
    adj = Variable(torch.Tensor(adj).cuda(gpu))

    ## applying gat
    output = model(cnn_set, adj, cnn=False)
    adj = adj.detach().cpu()
    cnn_set = cnn_set.detach().cpu()

    print('After GAT: ', output)
    torch.cuda.empty_cache()


    return output





def train_model(model, train_data, train_ims, train_adjs, val_data, val_ims, val_adjs, criterion, gpu='cuda:0', BATCH_SIZE=16, lr=1e-5, lr_step=5, lr_gamma=0.5, epochs=1000, patience=30, grad_clip=None):
    # empty list to store training losses
    loss_vals = []
    # empty list to store validation losses
    losses = []
    t_total = time.time()
    best = 1e10

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)   #1e-6
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    for epoch in range(epochs):        
        tr_loss = 0
        i = 0
        print('Current learning rate: ', scheduler.get_last_lr())
        train_losses = []

        for i, tr_dataset in enumerate(train_data):
            
            # clearing the Gradients of the model parameters
            optimizer.zero_grad()
            tr_output = single_sample_run(model, train_ims[i], tr_dataset, train_adjs[i], BATCH_SIZE=BATCH_SIZE, gpu=gpu)
            
            ## computing train loss for a single spatial sample
            tr_dataset = Variable(torch.Tensor(tr_dataset).cuda(gpu))
            loss_train = criterion(tr_output, tr_dataset)
            loss_train.backward()

            torch.cuda.empty_cache()
            tr_dataset = tr_dataset.detach().cpu()
            tr_output = tr_output.detach().cpu()
            loss_train = loss_train.detach().cpu()

            # Gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            tr_loss = loss_train.item()
            
            torch.cuda.empty_cache()
            train_losses.append(tr_loss)
            torch.cuda.empty_cache()
            
            if epoch%10 == 0:
                print("True trainset expression: ")
                print(tr_dataset)
                print("Predicted trainset expression: ")
                print(tr_output)

                temp_true = np.array(tr_dataset)
                temp_out = np.array(tr_output)
     
                corr_tr = np.corrcoef(temp_true, temp_out)
                print('Sample train mean correlation: ', corr_tr.mean())
            
            torch.cuda.empty_cache()
        scheduler.step()

        val_l,val_corr = validate_model(model, val_data, val_ims, val_adjs, criterion, gpu=gpu, BATCH_SIZE=BATCH_SIZE)
        loss_vals.append(val_l)

        print('Epoch : ',epoch, '\t', 'loss :', np.mean(train_losses))
        print('val_loss: ', val_l, 'val mean correlation: ', val_corr)
        torch.cuda.empty_cache()

        os.makedirs('saved_model/', exist_ok=True)
        torch.save(model.state_dict(), 'saved_model/{}.pkl'.format(epoch))
        if loss_vals[-1] < best:
            best = loss_vals[-1]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == patience:
            break

        files = glob.glob('saved_model/*.pkl')
        for file in files:
            epoch_nb = int(file.split('/')[1].split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(file)

        print('----------------------------------------------------------------------------------------')


    files = glob.glob('saved_model/*.pkl')
    for file in files:
        epoch_nb = int(file.split('/')[1].split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(file)

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Restore best model
    print('Loading {}th epoch'.format(best_epoch))
    model.load_state_dict(torch.load('saved_model/{}.pkl'.format(best_epoch)))
    
    return model







def validate_model(model, val_data, val_ims, val_adjs, criterion, gpu='cuda:0', BATCH_SIZE=16):
    # empty list to store test losses
    val_losses = []
    val_corrs = []

    model.eval()
    
    for i, val_dataset in enumerate(val_data):

        val_output = single_sample_run(model, val_ims[i], val_dataset, val_adjs[i], BATCH_SIZE=BATCH_SIZE, gpu=gpu)
        
        ## computing train loss for a single spatial sample
        val_dataset = torch.Tensor(val_dataset).cuda(gpu)
        loss_val = criterion(val_output, val_dataset)

        val_dataset = val_dataset.detach().cpu()
        val_output = val_output.detach().cpu()
        loss_val = loss_val.detach().cpu()

        val_loss = loss_val.item()
        
        torch.cuda.empty_cache()
        val_losses.append(val_loss)

        temp_true = np.array(val_dataset)
        temp_out = np.array(val_output)

        corr_val = np.corrcoef(temp_true, temp_out)
        val_corrs.append(corr_val.mean())

        torch.cuda.empty_cache()

    return np.mean(val_losses), np.mean(val_corrs)








def test_model(model, test_data, test_ims, test_adjs, test_names, genes, root_dir, gpu='cuda:0', BATCH_SIZE=16):
    print('-----------------------------Testing--------------------------------------')
    # empty list to store test losses
    test_losses = []
    test_corrs = []

    model.eval()

    for i, test_dataset in enumerate(test_data):
        print('---------------')
        ts_output = single_sample_run(model, test_ims[i], test_dataset, test_adjs[i], BATCH_SIZE=BATCH_SIZE, gpu=gpu)
        
        tl_func = nn.MSELoss()
        
        ## computing train loss for a single spatial sample
        ts_output = ts_output.detach().cpu()
        test_dataset = torch.Tensor(test_dataset)    
        loss_test = tl_func(ts_output, test_dataset)

        ts_loss = loss_test.item()
        torch.cuda.empty_cache()
        test_losses.append(ts_loss)

        print("True testset expression: ")
        print(test_dataset)
        print("Predicted testset expression: ")
        print(ts_output)

        temp_true = pd.DataFrame(np.array(test_dataset))
        temp_out = pd.DataFrame(np.array(ts_output))

        temp_true = pd.DataFrame(np.array(test_dataset))
        temp_out = pd.DataFrame(np.array(ts_output))
        torch.cuda.empty_cache()

        corr_ts = temp_out.corrwith(temp_true, axis=1)
        test_corr = corr_ts.mean()
        test_corrs.append(test_corr)
        print('test single dataset correlation: ', test_corr)

        print(corr_ts)
        temp_file = pd.read_csv(root_dir + '/gene_exp/' + test_names[i]+'_exp.csv')
        spot_names = temp_file.index
        
        corr_ts.index = spot_names
        print(corr_ts)
        os.makedirs('./results/', exist_ok=True)
        corr_ts.to_csv('./results/'+test_names[i]+'_corr.csv')


        temp_out = pd.DataFrame(temp_out)
        temp_out.index = spot_names
        temp_out.columns = genes
        temp_out.to_csv('./results/'+test_names[i]+'_out.csv')

        torch.cuda.empty_cache()

    
    print('Test Corr: ', np.mean(test_corrs))
    print('Test Loss: ', np.mean(test_losses))
    print('----------------------------------------------------------------------------------------')

    return test_corr





def load_data(names, genes, root_dir, patch_dir, adj_threshold):
    
    exp_dir = root_dir + '/gene_exp/'
    coord_dir = root_dir + '/coords/'


    first = True
    sample_nums = []
    adj_matrices = []
    sample_names = []
    Sample_images = []

    for f in names:
        f_name = f.split('.')[0]
        print('\nLoading sample: ', f_name)
        img_dir = patch_dir + f_name + '/'
        
        exp = pd.read_csv(exp_dir + f_name + '_exp.csv', index_col=0)
        
        Sample_images.append(load_spot_images(img_dir, exp.index))
        coord_file_name = coord_dir + f_name + '_coords.csv'
        coords = pd.read_csv(coord_file_name, index_col=0)
        adj_matrices.append(create_adj(coords.loc[exp.index], adj_threshold))

        inds = []
        for (j,name) in enumerate(exp.index):
            inds.append(f_name + '-' + name)
            
        exp.index = inds
        exp = exp[genes]

        if first: 
            raw_counts = exp
            first = False
        else: 
            raw_counts = pd.concat([raw_counts, exp])

        print('Sample name: ' + f + ' number of spots: ' + str(exp.shape[0]))
        sample_nums.append(exp.shape[0])
        sample_names.append(f_name)


    samples = pd.Series(sample_nums, index=sample_names)               

    data = process_separate(raw_counts, samples, genes)
    return data, adj_matrices, Sample_images, sample_names



## train and test SEG module
def run_SEG(root_dir, patience, adj_threshold, nb_heads, nb_embed, n_epochs, lr, lr_step, lr_gamma, BATCH_SIZE, gpu='cuda:0'):
    
    patch_dir = root_dir + '/patches/'
    
    genes = pd.read_csv(root_dir + '/spatial_gene_names.csv', index_col=0)['Gene']
    n_features = genes.shape[0]  


    names = os.listdir(patch_dir)
    tr_x,test_names,__,_=train_test_split(names,
                    names,test_size=0.3,random_state=seed)

    train_names,val_names,_,_=train_test_split(tr_x,
                    __,test_size=0.3,random_state=seed)


    # genes = pd.read_csv('../GC_data/gc_32_genes.csv', index_col=0)
    train_data, train_adjs, train_ims, _ = load_data(train_names, genes, root_dir, patch_dir, adj_threshold)
    test_data, test_adjs, test_ims, test_names = load_data(test_names, genes, root_dir, patch_dir, adj_threshold)
    val_data, val_adjs, val_ims, _ = load_data(val_names, genes, root_dir, patch_dir, adj_threshold)

    n_features = test_data[0].shape[1]
    print('Number of features: ', n_features)

    ## freeing up cache
    torch.cuda.empty_cache()

    # defining the model
    model = SEG(in_channels = 1, nb_genes = n_features, nb_heads = nb_heads, nb_embed=nb_embed)
    # clipper = WeightClipper()
    # model.apply(clipper)


    # defining the loss function
    criterion = nn.MSELoss()


    # checking if GPU is available
    if torch.cuda.is_available():
        model = model.to(gpu)
        criterion = criterion.cuda(gpu)
    print(model)


    # training
    model = train_model(model, train_data, train_ims, train_adjs, val_data, val_ims, val_adjs, criterion, gpu=gpu, BATCH_SIZE=BATCH_SIZE, lr=lr, lr_step=lr_step, lr_gamma=lr_gamma, epochs=n_epochs, patience=patience, grad_clip=None)
    os.makedirs('./trained/', exist_ok=True)
    

    # testing
    test_model(model, test_data, test_ims, test_adjs, test_names, genes, root_dir, gpu=gpu, BATCH_SIZE=BATCH_SIZE)

