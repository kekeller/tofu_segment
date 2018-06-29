
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
from skimage.filters import roberts, sobel, scharr, prewitt


from skimage import io; io.use_plugin('matplotlib')

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

	x_16 = []
	x_8 = []
	x_4 = []
	edges = []
	y = []
	original_image = []
	for i in range(len(train)):
		name = imgDir + str(train[i]) + '.jpg'

		image = io.imread(name,as_gray=True)
		edge_sobel = sobel(image)
		edges.append(edge_sobel.ravel())

		image = io.imread(name)
		original_image.append(image)

		fd, hog_image = hog(  filters.gaussian(image), orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)
		x_16.append( fd)

		fd, hog_image = hog(  filters.gaussian(image), orientations=8, pixels_per_cell=(8, 8),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)
		x_8.append( fd ) 

		fd, hog_image = hog(  filters.gaussian(image), orientations=8, pixels_per_cell=(4, 4),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)
		x_4.append( fd ) 

		name = maskDir + str(train[i]) + '.png'
		mask =  img_as_uint(io.imread(name).ravel())

		y.append(mask)

	return x_16,x_8,x_4,y,original_image, edges

def train_model(train_x,train_y):
	print("start train model")
	# Create random forest classifier instance
	trained_model = RandomForestClassifier(verbose=1)
	trained_model.fit(train_x, train_y)
	print("Trained model :: " +str(trained_model) )

	s = joblib.dump(trained_model, 'model.pkl',compress=9) 
	print("model saved")

def pred_model(test_x, test_y,original):
	print("start load model")
	trained_model = joblib.load('model.pkl')
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

	for i in range(len(predictions)):
		io.imsave(str(i) + "_out_pred.tif", img_as_float(predictions[i].reshape(256,256,1)))
		io.imsave(str(i) + "_out_img.tif", img_as_uint(original[i].reshape(256,256,3)))


def cal_miou(pred,val):

	tn, fp, fn, tp = confusion_matrix(val,pred).ravel()

	miou = tp / float(tp + fp + fn)
	print(miou)
	return miou

## load data
imgDir = './data_small/train_images/'
maskDir = './data_small/train_masks/'

imgValDir = './data_small/validate_images/'
maskValDir = './data_small/validate_masks/'

train_x_16,train_x_8,train_x_4,train_y,train_orig,train_edges = read_data(imgDir, maskDir)
test_x_16,test_x_8,test_x_4, test_y,test_orig, test_edges = read_data(imgValDir, maskValDir)

x_stack = []

print(len(train_edges[0]))


for x16,x8,x4,edge in zip(train_x_16,train_x_8,train_x_4,train_edges):
	m = np.concatenate([  x16, x8, x4,edge]) #, orig[:,:,0].ravel(), orig[:,:,1].ravel(), orig[:,:,2].ravel()])
	x_stack.append(m)


x_stack_val = []

for x16,x8,x4,edge in zip(test_x_16,test_x_8,test_x_4,test_edges):
	m = np.concatenate([  x16, x8, x4,edge]) #, orig[:,:,0].ravel(), orig[:,:,1].ravel(), orig[:,:,2].ravel()])
	x_stack_val.append(m)


print("train set: " + str(len(train_orig)))
print("test set: " + str(len(test_orig)))

print(len(x_stack))
print(len(x_stack[0]))
print(len(train_y))

train_model(x_stack, train_y)
pred_model(x_stack_val, test_y, test_orig)