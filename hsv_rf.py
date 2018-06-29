from sklearn.ensemble import RandomForestClassifier
import glob
from skimage import io,img_as_uint, img_as_float
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_squared_error, jaccard_similarity_score, confusion_matrix
from sklearn.externals import joblib
from skimage.feature import hog
from skimage.color import rgb2gray 
import matplotlib.pyplot as plt
from skimage import data, exposure, filters

from skimage import io; io.use_plugin('matplotlib')
from skimage.color import rgb2hsv

import matplotlib.cm as cm

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

	x = []
	y = []
	original = []
	for i in range(len(train)):
		name = imgDir + str(train[i]) + '.jpg'
		image = io.imread(name)
		original.append(image)

		#fd, hog_image = hog(  image, orientations=8, pixels_per_cell=(2, 2),
        #            cells_per_block=(1, 1), visualize=True, multichannel=True)

		hsv = filters.gaussian(image,sigma=3)
		hsv = rgb2hsv(hsv)
		#m = np.concatenate([image.ravel(),hsv.ravel()])

		#m = np.concatenate([hsv.ravel(),fd])

		#m = hsv.ravel()

		x.append(hsv.ravel())

		name = maskDir + str(train[i]) + '.png'
		mask =  img_as_uint(io.imread(name))
		print(np.mean(mask))
		print(np.unique(mask))
		mask = np.where(mask > 0.5, 1, 0) # convert to binary target pixel values
		print(np.mean(mask))
		print(np.unique(mask))
		print()
		y.append(mask.ravel())

	return x,y,original



def train_model(train_x, train_y, test_x, test_y):
	print("start train model")
	# Create random forest classifier instance
	# trained_model = RandomForestClassifier(verbose=1, n_estimators=1,warm_start=True,n_jobs=-1)

	# batch_size = 4
	# split = len(train_x) / batch_size 
	# print("Split data to: " + str(split))
	# step = 0

	# while (step < split):
	# 	print("Step number: " + str(step))
	# 	trained_model.n_estimators = trained_model.n_estimators + 4
	# 	trained_model.fit(train_x[step*batch_size : (step+1)*batch_size], train_y[step*batch_size : (step+1)*batch_size])
	# 	print("length of batch: " + str(len(train_x[step*batch_size : (step+1)*batch_size])))
	# 	step += 1

	trained_model = RandomForestClassifier(verbose=1, n_estimators=4,warm_start=True, n_jobs=-1)

	batch_size = 2
	split = len(train_x) / batch_size + 1
	print("Split data to :" + str(split-1))
	step = 1

	while (step < split):
		print("Step number: " + str(step - 1))
		print("length of batch: " + str(len(train_x[batch_size*(step-1):step*batch_size])))
		trained_model.fit(train_x[batch_size*(step-1):step*batch_size], train_y[batch_size*(step-1):step*batch_size])
		trained_model.n_estimators = trained_model.n_estimators + 4
		step += 1

	trained_model.n_estimators = trained_model.n_estimators - 4
	print("Trained model :: " +str(trained_model) )

	predictions = trained_model.predict(test_x[0])
	pred = []

	for p in predictions:
		pred.append(np.array(p,dtype=int))

	print(predictions)
	print(np.mean(predictions[0]))
	print(np.mean(test_y[0]))

	miou = 0
	for p,y in zip(pred,test_y):
		miou = miou + cal_miou(p,y) 

	miou = miou / len(pred)
	print("mIOU :: " + str(miou))

	s = joblib.dump(trained_model, 'model.pkl',compress=9) 
	print("model saved")

def pred_model(test_x, test_y,original_images):
	print("start load model")
	trained_model = joblib.load('model_hsv_rf_240.pkl')
	print("model loaded")
	predictions = trained_model.predict(test_x)
	pred = []

	for p in predictions:
		pred.append(np.array(p,dtype=int))

	miou = 0
	for p,y in zip(pred,test_y):
		miou = miou + cal_miou(p,y) 

	miou = miou / len(pred)
	print("mIOU :: " + str(miou))

	print(predictions[0][0])

	for i in range(len(predictions)):
		print(i)
		io.imsave("./out/" + str(i) + "_pred.tif", img_as_float(predictions[i].reshape(256,256,1)))
		io.imsave("./out/" + str(i) + "_img.tif", img_as_uint(original_images[i]) )

		#plt.imsave("./out/" + str(i) + "_mask.png", np.array(test_y[i].reshape(256,256,1)), cmap=cm.gray)
		# plt.imsave("./out/" + str(i) + "_img.png", np.array(test_x[i].reshape(256,256,3)), cmap=cm.gray)

		# background = Image.open("./out/" + str(i) + "_pred.png")
		# overlay = Image.open("./out/" + str(i) + "_img.png")

		# background = background.convert("RGBA")
		# overlay = overlay.convert("RGBA")

		# new_img = Image.blend(background, overlay, 0.5)
		# new_img.save("./out/" + str(i) + "_new.png","PNG")

def cal_miou(pred,val):

	tn, fp, fn, tp = confusion_matrix(val,pred).ravel()

	miou = tp / float(tp + fp + fn)
	print(miou)
	return miou

## load data
imgDir = './data_small/train_images/'
maskDir = './data_small/train_masks/'

imgValDir = './data_full/validate_images/'
maskValDir = './data_full/validate_masks/'

train_x,train_y,_ = read_data(imgDir, maskDir)
test_x, test_y,original_images = read_data(imgValDir, maskValDir)

print("train set: " + str(len(train_x)))
print("test set: " + str(len(test_x)))

print(len(train_x[0]))
print(len(train_x))

train_count = 8
print("Training on: " + str(train_count))

print(train_x[0:train_count])
print(train_y[0:train_count])
print(test_x)
print(test_y)
print

batch_size = 4
step = 1
print(train_x[batch_size*(step-1):step*batch_size])
print(train_y[batch_size*(step-1):step*batch_size])
step = 2
print(train_x[batch_size*(step-1):step*batch_size])

#train_model(train_x[0:train_count],train_y[0:train_count],test_x, test_y)
pred_model(test_x, test_y,original_images)