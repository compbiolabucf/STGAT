import cv2
import os
import pandas as pd



def generate_patch(patch_dir, image, image_name, x_coord, y_coord, names, spot_size=224):
    
    half_size = int(spot_size/2)
    for i in range(x_coord.shape[0]):
        print("Saving patch at coord: ", x_coord[i], y_coord[i])

        patch = image[x_coord[i]-half_size:x_coord[i]+half_size, y_coord[i]-half_size:y_coord[i]+half_size]
        patch_img_dir = patch_dir + image_name + '/'
        os.makedirs(patch_img_dir, exist_ok=True)

        patch = cv2.resize(patch, (224, 224), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(patch_img_dir + names[i] + ".jpg", patch)


def split_spatial(root_dir, spot_size=224):
    img_dir = root_dir + '/wsi/'
    coord_dir = root_dir + '/coords/'
    gene_exp_dir = root_dir + '/gene_exp/'
    patch_dir = root_dir + '/patches/'
    os.makedirs(patch_dir, exist_ok=True)
    
    for image_name in os.listdir(img_dir):
        # print(image_name)
        img = cv2.imread(img_dir+image_name)
        img_name = image_name.split('.')[0]
        print("Reading image: ", img_name)

        coords = pd.read_csv(coord_dir+img_name+'_coords.csv', index_col=0)
        gene_exp = pd.read_csv(gene_exp_dir+img_name+'_exp.csv', index_col=0)
        coords = coords.loc[gene_exp.index]
        # coords.drop(['in_tissue'], axis=1, inplace=True)
        coords = coords.round(0).astype(int)
        # coords.columns = ['pixel_x', 'pixel_y']
        x_coord = coords['X']
        y_coord = coords['Y']

        generate_patch(patch_dir, img, img_name, x_coord, y_coord, gene_exp.index, spot_size)