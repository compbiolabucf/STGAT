import cv2
import os
import numpy as np
import slideio
import pandas as pd


def show_slideimage(image):
	cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('frame', 900, 900)
	cv2.imshow('frame', image)



def save_patches(image, out_path, csv_out_path, spot_size=512):

	os.makedirs(out_path, exist_ok=True)
	rows = int(image.shape[0]/spot_size) + 1
	cols = int(image.shape[1]/spot_size) + 1

	print(rows, cols)
	x_coord = []
	y_coord = []
	spot_names = []

	for r in range(rows):
		for c in range(cols):
			print(r, c)
			patch = image[r*spot_size:(r+1)*spot_size, c*spot_size:(c+1)*spot_size]
			mean_RGB = patch[:,:, 0]/3 + patch[:,:, 1]/3 + patch[:,:, 2]/3

			p_cnt = np.count_nonzero(np.where(mean_RGB < 220))

			if(p_cnt > spot_size*spot_size / 2):
				x_coord.append(r*512+224)
				y_coord.append(c*512+224)
				spot_names.append(str(r)+'x'+str(c))
				
				patch = cv2.resize(patch, (224, 224), interpolation=cv2.INTER_CUBIC)
				cv2.imwrite(out_path + str(r) + "x" + str(c) + ".jpg", patch)

	pd.DataFrame(data={'X': x_coord, 'Y':y_coord}, index=spot_names).to_csv(csv_out_path + '.csv')





def split_tcga(root_dir, spot_size=512):
	
	tcga_dir = root_dir + '/wsi/'
	patch_dir = root_dir + '/patches/'	
	coord_dir = root_dir + '/coords/'
 
	for name in os.listdir(tcga_dir):
     
		path = tcga_dir + name 
		patch_out_path = patch_dir + path.split("/")[-1][:-4] + "/"
		csv_out_path = coord_dir + path.split("/")[-1][:-4]
		slide = slideio.open_slide(path,'SVS')
  
		# num_scenes = slide.num_scenes
		scene = slide.get_scene(0)
		image = scene.read_block()
		
		# x_len = image.shape[1]
		save_patches(image, patch_out_path, csv_out_path, spot_size)
