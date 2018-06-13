import glob
import numpy as np
import os
from PIL import Image, ImageOps

maskDir =  '/home/kevin/Documents/Data_Set/LabeledData/augmented_data/validate_masks/'  #'./data/train_masks/'
flipDir = '/home/kevin/Documents/Data_Set/LabeledData/augmented_data/validate_masks/augment/'

# return a list of images 
def flip_image(maskDir, flipDir):
	images = glob.glob(maskDir + "*.png")
	print(images)

	for img in images:

		print(img)
		mask =  Image.open(img)
		maskFlip = ImageOps.flip(mask)

		name = img.replace(maskDir,flipDir)

		name = name.replace('.png', '_FLIP.png')
		maskFlip.save(name)


		maskFlip = ImageOps.mirror(mask)
		name = img.replace(maskDir,flipDir)

		name = name.replace('.png', '_MIRROR.png')
		maskFlip.save(name)

		maskFlip = mask.rotate(90)
		
		name = img.replace(maskDir,flipDir)
		name = name.replace('.png', '_ROTATE90.png')
		maskFlip.save(name)


if not (os.path.exists(flipDir)):
	os.mkdir(flipDir)

flip_image(maskDir,flipDir)