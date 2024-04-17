import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import torch.nn.functional as F
import os
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix

# for reading and displaying images
from skimage.io import imread
# get_ipython().run_line_magic('matplotlib', 'inline')

# for evaluating the model
from tqdm import tqdm

# PyTorch libraries and modules
import torch
from torch import nn

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
gpu = 'cuda:0'




def get_coord(raw_counts):
    X_coord = raw_counts['X']
    Y_coord = raw_counts['Y']
    
    X_coord = np.array(X_coord.astype(float))
    Y_coord = np.array(Y_coord.astype(float))
    

    return X_coord, Y_coord




def create_adj(raw_counts, adj_threshold=0.2):
    x_coord, y_coord = get_coord(raw_counts)
    
    # creating the distance matrix
    dim = x_coord.shape[0]
    distance_mat = np.zeros([dim,dim])
    imdt = np.zeros([2,dim])
    for i in range(dim):
        imdt[0] = x_coord - x_coord[i]
        imdt[1] = y_coord - y_coord[i]
        imdt = imdt**2
        distance_mat[i] = np.sqrt(imdt[0]+imdt[1])
    
    # normalizing distance matrix
    for i in range(dim):
        distance_mat[i] = distance_mat[i]/max(distance_mat[i])
        
    ## adjacency matrix
    adj_mat = distance_mat.copy()
    adj_mat[distance_mat<=adj_threshold] = 1
    adj_mat[distance_mat>adj_threshold] = 0
    np.fill_diagonal(adj_mat, 0)
    print('Adjacency matrix density: ', adj_mat[adj_mat==1].shape[0]/(dim*dim))
    
    return adj_mat



def load_spot_images(im_dir, spot_names):
    # loading training images
    imgs = []
    for img_name in tqdm(spot_names):
        # defining the image path
        image_path = im_dir + str(img_name) + '.jpg'
        # reading the image
        img = imread(image_path, as_gray=True)
        # normalizing the pixel values
        # img /= 255.0
        # converting the type of pixel to float 32
        img = img.astype('float32')
        # appending the image into the list
        imgs.append(img)

    # converting the list to numpy array
    return np.array(imgs)
    





def process_separate(Sample_images, label_data):

    # raw_data = raw_data['tumor_status']
    # raw_data.replace(to_replace={'non', 'tumor'}, value={0,1}, inplace=True)
    
    patch_ind = 0
    data = []
    for image in Sample_images:
        num_patches = image.shape[0]
        im_data = []
        l_data = label_data[patch_ind]
        
        
        for i in range(num_patches):
            # print('l_data.iloc[{}]= '.format(i), l_data.iloc[i].item())
            im_data.append([image[i], l_data.iloc[i].item()])
        
        data.append(im_data)
        patch_ind += 1

    data = [item for sublist in data for item in sublist]
    np.random.shuffle(data)
    
    return data





## using a GPU
def get_default_device(gpu='cuda:0'):
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device(gpu)
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)





## image processing
class transform_dataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, ims, mn, sd):
        'Initialization'
        self.ims = ims
        self.transform_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mn, sd, inplace=True)
        ])
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.ims)
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        image = self.ims[index][0]
        X = self.transform_norm(image)
        return X, self.ims[index][1]
        
    





## preparing dataloader for a single spatial sample
def get_dataloader(image_set, BATCH_SIZE, gpu='cuda:0'):

    ## stats for normalization
    init_dataloader = DataLoader(image_set, batch_size=len(image_set), shuffle=True)
    data_for_stats, _  = next(iter(init_dataloader))
    mn = data_for_stats[:,:,:].mean()
    sd = data_for_stats[:,:,:].std()


    dataset = transform_dataset(image_set, mn, sd)
    device = get_default_device(gpu)

    return DeviceDataLoader(DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True), device)



def roc_score(output, labels):
    preds = output[:,1]
    return roc_auc_score(labels, preds)
    

def sense_spec(outputs, labels, threshold):
    preds = np.array([1 if p>threshold else 0 for p in outputs[:,1]])
    tn, fn, fp, tp = confusion_matrix(labels, preds).ravel()
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    specificity = tn/(tn+fp)
    # print('precision = {}\nrecall/sensitivity = {}\nspecificity = {}'.format(precision, recall, specificity))
    return recall, specificity



class F1_Loss(nn.Module):
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    '''
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true,):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, 2).to(torch.float32)
        # y_pred = F.softmax(y_pred, dim=1)
        
        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2* (precision*recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        return 1 - f1.mean()









# --------------------------- TCGA Processing -----------------------------


def load_tcga(im_dir):
    # loading training images
    # im_dir = '../tcga_test/tcga_patches/' + subtype + image_name + '/image_{}/'.format(img_ind)

    imgs = []
    spot_names = os.listdir(im_dir)
    for img_name in tqdm(spot_names):
        # defining the image path
        image_path = im_dir + str(img_name) #+ '.jpg'
        # reading the image
        img = imread(image_path, as_gray=True)
        # normalizing the pixel values
        # img /= 255.0
        # converting the type of pixel to float 32
        img = img.astype('float32')
        # appending the image into the list
        imgs.append(img)

    np.random.shuffle(imgs)
    # converting the list to numpy array
    return spot_names, np.array(imgs)




## image processing
class transform_tcga(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, ims, mn, sd):
        'Initialization'
        self.ims = ims
        self.transform_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mn, sd, inplace=True)
        ])
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.ims)
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        image = self.ims[index]
        X = self.transform_norm(image)
        return X
 



def tcga_loader(im_set, BATCH_SIZE, gpu='cuda:0'):
    init_dataloader = DataLoader(im_set, batch_size=im_set.shape[0], shuffle=True)
    data_for_stats  = next(iter(init_dataloader))
    mn = data_for_stats[:,:,:].mean()
    sd = data_for_stats[:,:,:].std()


    dataset = transform_tcga(im_set, mn, sd)
    device = get_default_device(gpu)

    return DeviceDataLoader(DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True), device)
