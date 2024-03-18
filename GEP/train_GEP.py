import time
import glob
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import torch
from torch.autograd import Variable
from torch import nn, optim

from .models import *
from .utils import *


seed = 0
np.random.seed(seed)
torch.manual_seed(seed)



def single_sample_run(model, tcga_exp, image_set, adj, BATCH_SIZE=16, gpu='cuda', eval=True):
    torch.cuda.empty_cache()
    loader = get_dataloader(image_set, BATCH_SIZE, gpu)

    cnn_set = Variable(torch.zeros(image_set.shape[0], 16*7*7).cuda(gpu))
    for i, batch in enumerate(loader):

        cnn_out = model(batch, adj, tcga_exp, cnn=True)
        torch.cuda.empty_cache()

        batch = batch.detach().cpu()
        cnn_out = cnn_out.detach().cpu()

        if (i+1)*BATCH_SIZE>=image_set.shape[0]:cnn_set[i*BATCH_SIZE:] = cnn_out
        else: cnn_set[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = cnn_out

    print('After only CNN: ', cnn_set)
    adj = Variable(torch.Tensor(adj).to(gpu))
    cnn_set = cnn_set.to(gpu)

    torch.cuda.empty_cache()

    ## applying gat
    output = model(cnn_set, adj, tcga_exp, cnn=False)
    adj = adj.detach().cpu()
    cnn_set = cnn_set.detach().cpu()
    tcga_exp = tcga_exp.detach().cpu()

    print('After GAT: ', output)
    torch.cuda.empty_cache()
    

    return output







def train_model(model, train_data, train_ims, train_adjs, val_data, val_ims, val_adjs, criterion, gpu='cuda', BATCH_SIZE=16, lr=1e-5, lr_step=5, lr_gamma=0.75, epochs=1000, patience=30, grad_clip=None):

    # empty list to store validation losses
    loss_vals = []
    t_total = time.time()
    best = 1e10

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)   #1e-6
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    for epoch in range(epochs):        
        tr_loss = 0
        i = 0
        print('Current learning rate: ', scheduler.get_last_lr())
        train_losses = []

        for i, name in enumerate(train_data.index):
            print(i,name)
            
            # clearing the Gradients of the model parameters
            optimizer.zero_grad()
            tr_data = Variable(torch.Tensor(train_data.loc[name]).to(gpu))
            tr_out = single_sample_run(model, tr_data, train_ims[i], train_adjs[i], BATCH_SIZE=BATCH_SIZE, gpu=gpu)
            tr_output = torch.mean(tr_out, dim=0)
  
            ## computing train loss for a single sample
            loss_train = criterion(tr_output, tr_data)
            loss_train.backward()
            torch.cuda.empty_cache()


            tr_data = tr_data.detach().cpu()
            tr_output = tr_output.detach().cpu()
            tr_out = tr_out.detach().cpu()
            loss_train = loss_train.detach().cpu()

            # Gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            tr_loss = loss_train.item()
            
            torch.cuda.empty_cache()
            train_losses.append(tr_loss)

            if epoch%10 == 0:
                print("True trainset expression: ")
                print(tr_data)
                print("Predicted trainset expression: ")
                print(tr_output)

                temp_true = np.array(tr_data)
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

        os.makedirs('./saved_model/', exist_ok=True)
        torch.save(model.state_dict(), './saved_model/{}.pkl'.format(epoch))
        if loss_vals[-1] < best:
            best = loss_vals[-1]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == patience:
            break

        files = glob.glob('./saved_model/*.pkl')
        for file in files:
            epoch_nb = int(file.split('/')[2].split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(file)

        print('----------------------------------------------------------------------------------------')


    files = glob.glob('./saved_model/*.pkl')
    for file in files:
        epoch_nb = int(file.split('/')[2].split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(file)

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Restore best model
    print('Loading {}th epoch'.format(best_epoch))
    model.load_state_dict(torch.load('./saved_model/{}.pkl'.format(best_epoch)))


    return model







def validate_model(model, val_data, val_ims, val_adjs, criterion, gpu='cuda:0', BATCH_SIZE=16):
    # empty list to store test losses
    val_losses = []
    val_corrs = []

    model.eval()
    
    for i, name in enumerate(val_data.index):

        v_data = torch.Tensor(val_data.loc[name]).cuda(gpu)
        val_out = single_sample_run(model, v_data, val_ims[i], val_adjs[i], 
                    BATCH_SIZE=BATCH_SIZE, gpu=gpu, eval=True)
        val_output = torch.mean(val_out, dim=0)

        ## computing loss for a single sample
        loss_val = criterion(val_output, v_data)

        v_data = v_data.detach().cpu()
        val_output = val_output.detach().cpu()
        val_out = val_out.detach().cpu()
        val_loss = loss_val.item()
        
        torch.cuda.empty_cache()
        val_losses.append(val_loss)

        temp_true = np.array(v_data)
        temp_out = np.array(val_output)
        
        corr_val = np.corrcoef(temp_true, temp_out)
        val_corrs.append(corr_val)
        
        torch.cuda.empty_cache()
        
    return np.mean(val_losses), np.mean(val_corrs)








def test_model(model, test_data, test_ims, test_adjs, samples, spot_names_lst, gene_names, criterion, root_dir, gpu='cuda:0', BATCH_SIZE=16):
    print('-----------------------------Testing--------------------------------------')
    # empty list to store test losses
    test_losses = []
    test_corrs = []

    model.eval()

    for i, name in enumerate(test_data.index):
        print('---------------')
        t_data = torch.Tensor(test_data.loc[name]).cuda(gpu)
        ts_out = single_sample_run(model, t_data, test_ims[i], test_adjs[i], 
                    BATCH_SIZE=BATCH_SIZE, gpu=gpu, eval=True)
        ts_output = torch.mean(ts_out, dim=0)
        

        ## computing loss for a single spatial sample
        loss_test = criterion(ts_output, t_data)

        t_data = t_data.detach().cpu()
        ts_output = ts_output.detach().cpu()
        ts_out = ts_out.detach().cpu()

        ts_loss = loss_test.item()
        
        torch.cuda.empty_cache()
        test_losses.append(ts_loss)

        print("True testset expression: ")
        print(t_data)
        print("Predicted testset expression: ")
        print(ts_output)

        temp_true = pd.DataFrame(np.array(t_data))
        temp_out = pd.DataFrame(np.array(ts_output))
        torch.cuda.empty_cache()

        spots_out = pd.DataFrame(np.array(ts_out), index=spot_names_lst[i])
        spots_out.columns = gene_names
        
        os.makedirs('./prediction/', exist_ok=True)
        spots_out.to_csv('./prediction/'+samples.index[i]+'spots_pred.csv')

        test_corr = temp_out.corrwith(temp_true)
        test_corrs.append(test_corr)
        print('test single dataset correlation: ', test_corr)
        
        torch.cuda.empty_cache()
        
        # reading TCGA tumor labels:
        clinical_data = pd.read_csv(root_dir + '/clinical/' + samples.index[i] + '_labels.csv', index_col=0)
        tumor_spots_inds = clinical_data.loc[clinical_data['TCGA_prediction']==1].index
        tumor_spots_out = spots_out.loc[tumor_spots_inds]
        tumor_spots_out.mean(axis=0).to_csv('./prediction/'+samples.index[i]+'bulk_tumor_exp.csv')
    
    print('Test Corr: ', np.mean(test_corrs))
    print('Test Loss: ', np.mean(test_losses))
    print('----------------------------------------------------------------------------------------')

    return test_corr





def load_data(names,im_dir,coord_dir, adj_threshold=0.2):

    sample_nums = []
    adj_matrices = []
    sample_names = []
    Sample_images = []
    spot_names_lst = []
    
    for name in names:
      
        img_dir = im_dir + name + '/'
        
        spot_names = os.listdir(img_dir)
        Sample_images.append(load_spot_images(spot_names, img_dir))
        adj_matrices.append(create_adj(coord_dir+name+'.csv', spot_names, adj_threshold))

        
        print('Sample name: ' + name + ' number of spots: ' + str(len(spot_names)))
        sample_nums.append(len(spot_names))
        sample_names.append(name)
        spot_names_lst.append(spot_names)

    samples = pd.Series(sample_nums, index=sample_names)               
    
    return adj_matrices, Sample_images, samples, spot_names_lst





#train and test GEP module
def run_GEP(root_dir, patience, adj_threshold, nb_heads, nb_embed, n_epochs, lr, lr_step, lr_gamma, BATCH_SIZE, gpu='cuda:0'):

    patch_dir = root_dir + '/patches/'
    coord_dir = root_dir + '/coords/'
    names = os.listdir(patch_dir)
        
    genes = pd.read_csv(root_dir + '../gene_names.csv', index_col=0)['Gene']
    n_features = genes.shape[0]  
    exp_data = pd.read_csv(root_dir + '/tcga_exp.csv',index_col=0)[genes]

    names = os.listdir(patch_dir)   #exp_data.index
    exp_data = exp_data.loc[names]

    # exp_data = np.log2(exp_data+1)
    tr_x,ts_x,train_data,test_data=train_test_split(names,
                    exp_data,test_size=0.2,random_state=seed)

    tr_x,val_x,train_data,val_data=train_test_split(tr_x,
                    train_data,test_size=0.2,random_state=seed)


    # print('train names: \n{}'.format(train_data))
    # print('val names: \n{}'.format(val_data))
    # print('test names: \n{}'.format(test_data))


    train_adjs, train_ims, _ , _ = load_data(tr_x, im_dir=patch_dir, coord_dir=coord_dir, adj_threshold=adj_threshold)
    test_adjs, test_ims, samples, spot_names_lst = load_data(ts_x, im_dir=patch_dir, coord_dir=coord_dir, adj_threshold=adj_threshold)
    val_adjs, val_ims, _ , _ = load_data(val_x, im_dir=patch_dir, coord_dir=coord_dir, adj_threshold=adj_threshold)
    print('Number of features: ', n_features)


    torch.cuda.empty_cache()

    # defining the model
    model = GEP(nb_genes = n_features, 
                    nb_heads = nb_heads, 
                    nb_embed = nb_embed,
                    gpu=gpu)
    criterion = nn.MSELoss()


    # moving model to GPU
    model = model.cuda(gpu)
    criterion = criterion.cuda(gpu)
    print(model)


    ### freezing the trained parts
    model.freeze()

    model = train_model(model, train_data, train_ims, 
                train_adjs, val_data, val_ims, val_adjs, 
                criterion, gpu=gpu, BATCH_SIZE=BATCH_SIZE, lr=lr, 
                lr_step = lr_step, lr_gamma = lr_gamma,
                epochs=n_epochs, patience=patience, 
                grad_clip=None)


    # testings
    test_model(model, test_data, test_ims, test_adjs, samples, spot_names_lst, genes, 
               criterion, root_dir, gpu=gpu, BATCH_SIZE=BATCH_SIZE)


    torch.save(model.state_dict(), './trained/trained_gep.pkl')

