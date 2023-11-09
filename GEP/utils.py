import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.io import imread

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


seed = 0
np.random.seed(seed)
torch.manual_seed(seed)





class WeightClipper(object):

    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(100,10000)
            module.weight.data = w




def get_coord(coords):
    X_coord = coords['X']
    Y_coord = coords['Y']
    
    X_coord = np.array(X_coord.astype(float))
    Y_coord = np.array(Y_coord.astype(float))
    
    return X_coord, Y_coord




def create_adj(coord_dir, spot_names, adj_threshold=0.2):
    spot_names = [name.split('.')[0] for name in spot_names]
    coord_file = pd.read_csv(coord_dir, index_col=0)
    coord_file = coord_file.loc[spot_names]
    # x_coord = coord_file['X']
    # y_coord = coord_file['Y']
    x_coord, y_coord = get_coord(coord_file)
    
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

    np.save('adj_mat.npy', adj_mat)
    np.save('dist_mat.npy', distance_mat)
    
    return adj_mat



def load_spot_images(spot_names, im_dir):
    # loading training images
    imgs = []
    # spot_names = os.listdir(im_dir)
    
    for img_name in tqdm(spot_names):
        # defining the image path
        image_path = im_dir + str(img_name) #+ '.jpg'
        # reading the image
        img = imread(image_path, as_gray=True)

        # converting the type of pixel to float 32
        img = img.astype('float32')
        # appending the image into the list
        imgs.append(img)

    # converting the list to numpy array
    return np.array(imgs)





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
        image = self.ims[index]
        X = self.transform_norm(image)
        return X
        
    





## preparing dataloader for a single spatial sample
def get_dataloader(image_set, BATCH_SIZE, gpu='cuda'):

    ## stats for normalization
    init_dataloader = DataLoader(image_set, batch_size=image_set.shape[0], shuffle=True)
    data_for_stats = next(iter(init_dataloader))
    mn = data_for_stats[:,:,:].mean()
    sd = data_for_stats[:,:,:].std()
    
    dataset = transform_dataset(image_set, mn, sd)
    device = get_default_device(gpu)

    return DeviceDataLoader(DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True), device)


    


# def mse_loss(a,b):
#     ss = []
#     for i in range(a.shape[0]):
#         ss.append((a[i]-b[i])**2)
#     return np.mean(np.array(ss))



# def process_separate(raw_data, samples, tumor_status='tumor'):

#     raw_data = raw_data.loc[raw_data['tumor_status']!=tumor_status]
#     raw_data.drop(['X','Y','tumor_status'], axis = 1, inplace=True)

#     gene_names = pd.read_csv( '../all_gene_names.csv', index_col=0, header=None).index
#     raw_data = raw_data[gene_names]

#     print('data shape: ', raw_data.shape)

#     current_index = 0
#     data = []
#     for ind, sample_number in enumerate(samples[:]):
#         data.append(np.array(raw_data[current_index : current_index + sample_number]))
#         current_index += sample_number
    
#     return data