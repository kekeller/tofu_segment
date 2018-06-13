import glob
from skimage import io,img_as_uint, img_as_float
import numpy as np


# return a list of images 
def read_data(imgDir, maskDir):
	images = glob.glob(imgDir + "*.jpg")
	masks = glob.glob(maskDir + "*.png")

	train = []
	label = []

	for im in images:
		n = im.replace(imgDir,'')
		n = n.replace('.jpg','')
		train.append(n)

	for mask in masks:
		n = mask.replace(maskDir,'')
		n = n.replace('.png','')
		label.append(n)

	if ( set(label) == set(train)):
		print("each image has mask")

	file = open('val.txt',"w")
	for name in label:
		n = imgDir + name + ".jpg "
		n = n.replace('./data','')
		file.write(n)
		print(n)

		n = maskDir + name + ".png"
		n = n.replace('./data','')
		file.write(n + "\n")


# imgValDir = './data/validate_images/'
# maskValDir = './data/validate_masks/'

imgDir =  '/home/kevin/Documents/Data_Set/LabeledData/augmented_data/data/train_images/' #'./data/train_images/'
maskDir =  '/home/kevin/Documents/Data_Set/LabeledData/augmented_data/data/train_masks/'  #'./data/train_masks/'

valImgDir = '/home/kevin/Documents/Data_Set/LabeledData/augmented_data/data/validate_images/' #'./data/validate_images/'
valMaskDir = '/home/kevin/Documents/Data_Set/LabeledData/augmented_data/data/validate_masks/'  #'./data/validate_masks/'

read_data(valImgDir,valMaskDir)