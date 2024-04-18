import os
import torch
import argparse
from torch.autograd import Variable
from torch import nn

from split.split_tcga import *
from SLP.apply_SLP import *
from GEP.models import *
from GEP.utils import *


seed = 0
np.random.seed(seed)
torch.manual_seed(seed)



parser = argparse.ArgumentParser()
parser.add_argument('--run_type', type=str, default='full', help='select if ST and TCGA')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--no_cuda', action='store_true', default=False)

## SEG and GEP parameters
parser.add_argument('--adj_threshold', type=float, default=0.2)
parser.add_argument('--nb_heads', type=int, default=12)
parser.add_argument('--nb_embed', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=16)

## SLP
parser.add_argument('--slp_batch_size', type=int, default=16)
parser.add_argument('--apply_dir', type=str, default='apply_data/input/')


args = parser.parse_args()
cuda = not args.no_cuda and torch.cuda.is_available()

if cuda: gpu='cuda'
else: gpu='cpu'


print("\n--------------Processing for TCGA data----------")
split_tcga(args.apply_tcga_dir)
print("--------Splitting done for TCGA data--------")

## Generating labels for TCGA data
apply_SLP(args.apply_dir, args.slp_batch_size)



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





def apply_model(model, test_data, test_ims, test_adjs, samples, spot_names_lst, gene_names, criterion, root_dir, gpu='cuda:0', BATCH_SIZE=16):
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
        
        os.makedirs('./apply_data/prediction/', exist_ok=True)
        spots_out.to_csv('./apply_data/prediction/'+samples.index[i]+'spots_pred.csv')

        test_corr = temp_out.corrwith(temp_true)
        test_corrs.append(test_corr)
        print('test single dataset correlation: ', test_corr)
        
        torch.cuda.empty_cache()
        
        # reading TCGA tumor labels:
        clinical_data = pd.read_csv(root_dir + '/clinical/' + samples.index[i] + '_labels.csv', index_col=0)
        tumor_spots_inds = clinical_data.loc[clinical_data['TCGA_prediction']==1].index
        tumor_spots_out = spots_out.loc[tumor_spots_inds]
        tumor_spots_out.mean(axis=0).to_csv('./apply_data/prediction/'+samples.index[i]+'bulk_tumor_exp.csv')
    
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




def run_model(root_dir, adj_threshold, nb_heads, nb_embed, BATCH_SIZE, gpu='cuda:0'):

    patch_dir = root_dir + '/patches/'
    coord_dir = root_dir + '/coords/'
        
    genes = pd.read_csv(root_dir + '../gene_names.csv', index_col=0)['Gene']
    n_features = genes.shape[0]  
    exp_data = pd.read_csv(root_dir + '/tcga_exp.csv',index_col=0)[genes]

    names = os.listdir(patch_dir)   #exp_data.index
    exp_data = exp_data.loc[names]

    # print('train names: \n{}'.format(train_data))
    # print('val names: \n{}'.format(val_data))
    # print('test names: \n{}'.format(test_data))


    adjs, ims, samples, spot_names_lst = load_data(names, im_dir=patch_dir, coord_dir=coord_dir, adj_threshold=adj_threshold)
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

    ## loading trained model
    model.load_state_dict(torch.load('./trained/{}.pkl'.format('trained_gep'), map_location=torch.device(gpu)))

    # applying model
    apply_model(model, exp_data, ims, adjs, samples, spot_names_lst, genes, 
               criterion, root_dir, gpu=gpu, BATCH_SIZE=BATCH_SIZE)


### calling the function for applying STGAT framework
run_model(args.apply_dir, args.adj_threshold, args.nb_heads, args.nb_embed, args.batch_size, gpu=gpu)