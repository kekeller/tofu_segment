from sklearn.ensemble import RandomForestClassifier
import glob
from skimage import io,img_as_uint, img_as_float
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_squared_error, jaccard_similarity_score
from sklearn.externals import joblib
from skimage.color import rgb2hsv


imgDir = "/home/kevin/Documents/Data_Set/data_960_aug_HSV/validate_images/"

images = glob.glob(imgDir + "*.tif")
print(images)


for img in images:
	image = io.imread(img)
	image = rgb2hsv(image)
	name = img.replace('tif','png')
	io.imsave(name,image)
