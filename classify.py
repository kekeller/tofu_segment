from sklearn.ensemble import RandomForestClassifier
import glob
from skimage import io,img_as_uint, img_as_float
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_squared_error
from sklearn.externals import joblib


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

	file = open('training.txt',"w")
	for name in label:
		file.write(imgDir + name + ".tif ")
		file.write(maskDir + name + ".png\n")


	x = []
	y = []
	for i in range(len(train)):
		name = imgDir + str(train[i]) + '.tif'
		x.append( img_as_float(io.imread(name).flatten() ) )

		name = maskDir + str(train[i]) + '.png'
		mask =  io.imread(name).flatten() 
		mask = np.where(mask > 0.5, 1, 0) # convert to binary target pixel values
		y.append(mask)

	return x,y

def train_model(train_x,train_y):
	print("start train model")
	# Create random forest classifier instance
	trained_model = RandomForestClassifier(verbose=1)
	trained_model.fit(train_x, train_y)
	print("Trained model :: " +str(trained_model) )

	s = joblib.dump(trained_model, 'model.pkl') 
	print("model saved")

def pred_model(test_x, test_y):
	print("start load model")
	trained_model = joblib.load('model.pkl')
	print("model loaded")
	predictions = trained_model.predict(test_x)

	print("Test mean squared error :: " + str(mean_squared_error(test_y,predictions)) )
	
	for i in range(len(predictions)):
		io.imsave(str(i) + "_out_pred.tif", img_as_uint(predictions[i].reshape(256,256,1)))
		io.imsave(str(i) + "_out_img.tif", img_as_uint(test_x[i].reshape(256,256,3)))


## load data
imgDir = './data/train_images/'
maskDir = './data/train_masks/'

imgValDir = './data/validate_images/'
maskValDir = './data/validate_masks/'

train_x,train_y = read_data(imgDir, maskDir)
#test_x, test_y = read_data(imgValDir, maskValDir)

print("train set: " + str(len(train_x)))
#print("test set: " + str(len(test_x)))

#train_model(train_x,train_y)
#pred_model(test_x, test_y)