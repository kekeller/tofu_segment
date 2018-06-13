from sklearn.ensemble import RandomForestClassifier
import glob
from skimage import io,img_as_uint, img_as_float
import numpy as np

from PIL import Image

# imgDir = './data/validate_images/'
# maskDir = './data/validate_masks/'

maskDir =  '/home/kevin/Documents/Data_Set/LabeledData/augmented_data/train_masks/'  #'./data/train_masks/'

# return a list of images 
def binary_mask_conv(maskDir):
	masks = glob.glob(maskDir + "*.png")

	label = []

	for mask in masks:
		n = mask.replace(maskDir,'')
		n = n.replace('.png','')
		label.append(n)

	for i in range(len(label)):

		name = maskDir + str(label[i]) + '.png'
		mask =  Image.open(name)

		print(np.array(mask).shape)
		mask = np.array( mask, dtype=np.uint8 )

		mask = np.where(mask > 0.5, 1, 0) # convert to binary target pixel values
		print(np.unique(mask))
		print(np.mean(mask))

		d2_array = np.zeros( (256,256) ,dtype=np.uint8)

		for i in range(256):
			for j in range(256):
				if (mask[i][j] == 0):
					d2_array[i][j] = 1

		
		mask = Image.fromarray(d2_array)
		mask.save(name)


binary_mask_conv(maskDir)


#convert *.tif -set filename: "%t" %[filename:].jpg