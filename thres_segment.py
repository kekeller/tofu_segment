from skimage import data
from skimage import filters
from skimage import exposure
from skimage import io,img_as_uint, img_as_float, filters
from skimage.color import rgb2hsv
from skimage.filters import threshold_local, threshold_otsu, rank, try_all_threshold
from skimage.morphology import disk

from sklearn.ensemble import RandomForestClassifier
import glob
from skimage import io,img_as_uint, img_as_float
import numpy as np
from sklearn.metrics import accuracy_score,mean_squared_error, jaccard_similarity_score, confusion_matrix

import matplotlib.pyplot as plt

import matplotlib.cm as cm
import numpy as np

import Image


# return a list of images 
def read_data(imgDir, maskDir):
	images = glob.glob(imgDir + "*.tif")
	masks = glob.glob(maskDir + "*.png")

	train = []
	label = []

	for im in images:
		n = im.replace(imgDir,'')
		n = n.replace('.tif','')
		train.append(n)

	for mask in masks:
		n = mask.replace(maskDir,'')
		n = n.replace('.png','')
		label.append(n)

	if ( set(label) == set(train)):
		print("each image has mask")

	x = []
	y = []
	for i in range(len(train)):
		name = imgDir + str(train[i]) + '.tif'
		x.append( io.imread(name) )

		name = maskDir + str(train[i]) + '.png'
		mask =  io.imread(name)
		mask = 1-mask
		y.append(mask)

	return x,y


def pred_mask(image):
	#image = filters.gaussian(image, sigma=3)
	image = rgb2hsv(image)
	image = image[:,:,0]
	val = filters.threshold_otsu(image)
	mask = image < val
	mask = mask*1

	return mask

def pred_model(test_x, test_y):

	predictions = []
	for image in test_x:
		predictions.append(pred_mask(image))

	# for i in range(len(predictions)):
	# 	plt.imsave("./out/" + str(i) + "_pred.png", np.array(predictions[i]), cmap=cm.gray)
	# 	plt.imsave("./out/" + str(i) + "_mask.png", np.array(test_y[i]), cmap=cm.gray)
	# 	plt.imsave("./out/" + str(i) + "_img.png", np.array(test_x[i]), cmap=cm.gray)

	# 	background = Image.open("./out/" + str(i) + "_pred.png")
	# 	overlay = Image.open("./out/" + str(i) + "_img.png")

	# 	background = background.convert("RGBA")
	# 	overlay = overlay.convert("RGBA")

	# 	new_img = Image.blend(background, overlay, 0.5)
	# 	new_img.save("./out/" + str(i) + "_new.png","PNG")

	miou = 0
	for p,y in zip(predictions,test_y):
		miou = miou + cal_miou(p.ravel(),y.ravel()) 

	miou = miou / len(predictions)
	print("mIOU :: " + str(miou))

def cal_miou(pred,val):

	tn, fp, fn, tp = confusion_matrix(val,pred).ravel()

	miou = float(tp) / float(tp + fp + fn)
	print(miou)


	return miou

## load data
imgDir = './data_small/train_images/'
maskDir = './data_small/train_masks/'

imgValDir = './data_tif/validate_images_(copy)/'
maskValDir = './data_tif/validate_masks_(copy)/'

#train_x,train_y = read_data(imgDir, maskDir)
test_x, test_y = read_data(imgValDir, maskValDir)

#print("train set: " + str(len(train_x)))
print("test set: " + str(len(test_x)))

pred_model(test_x, test_y)