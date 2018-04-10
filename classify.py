from sklearn.ensemble import RandomForestClassifier
import glob
from skimage import io,img_as_uint
from sklearn.neural_network import MLPClassifier
import numpy as np

# return a list of images 
def read_data(path):
	images = glob.glob(path + "/*.tif")

	train = []
	label = []
	for im in images:
		if "train" in im:
			label.append(im)
		if "train" not in im and "image" in im:
			train.append(im)

	# order images in set
	x = []
	for i in range(1,len(train)+1):
		for im in train:
			if "_"+str(i)+"." in im:
				x.append(im)
	train_list = x
	x = []
	for i in range(1,len(label)+1):
		for im in label:
			if "_"+str(i)+"." in im:
				x.append(im)
	label_list = x

	label  = []
	train = []
	for im in label_list:
		x = io.imread(im)
		label.append(x.flatten()) 
	for im in train_list:
		x = io.imread(im)
		train.append(x.flatten())

	return train,label

# return a black and white mask
def convert_label_mask(labels):
	mask = []
	for label in labels:
		m = []
		for x in label:
			if x > 150:
				m.append(np.uint8(255))
			else:
				m.append(np.uint8(0))

		m = np.array(m)
		m = m.reshape(250,250,3)

		x = []
		for row in m:
			for indx, pixel in enumerate(row):
				if np.mean(pixel) > 0:
					x = [np.uint8(255),np.uint8(255),np.uint8(255)]
					x = np.array(x)
					row[indx] = x
		mask.append(m.flatten())
		io.imsave("mask.tif",m.reshape(250,250,3))
	return mask

#train,label = read_data("./images")
#mask = convert_label_mask(label)

#train_validate,label_validate = read_data("./predict")
#mask_validate = convert_label_mask(label_validate)

def fake_images_green():
	im = np.linspace(0, 1., (250*250*3)).reshape(250,250,3)
	for row in im:
		for indx, pixel in enumerate(row):
			x = (0,1.0,0.255) # set to green
			x = np.array(x)
			row[indx] = x

	im = img_as_uint(im)
	return im

def fake_images_white():
	im = np.linspace(0, 1., (250*250*3)).reshape(250,250,3)
	for row in im:
		for indx, pixel in enumerate(row):
			x = (1,1,1) 
			x = np.array(x)
			row[indx] = x

	im = img_as_uint(im)
	return im

def fake_images_mask():
	im = np.linspace(0, 1., (250*250*3)).reshape(250,250,3)
	for row in im:
		for indx, pixel in enumerate(row):
			x = (0,0,0) 
			x = np.array(x)
			row[indx] = x

	im = img_as_uint(im)
	return im

def fake_data():
	green = fake_images_green()
	white = fake_images_white()
	mask = fake_images_mask()
	io.imsave("green.tif",green)
	io.imsave("white.tif",white)
	return green,white,mask

green,white,mask = fake_data()
x = []
x.append(green.flatten())
x.append(white.flatten())
y = []
y.append(mask.flatten())
y.append(white.flatten())
test = []
test.append(white.flatten())

clf = RandomForestClassifier(verbose=1)
clf.fit(x,y)
print(clf)
pred = clf.predict(test)
print(pred)

# clf.fit(train, mask)
# pred = clf.predict(train_validate)
# pred = np.array(pred)
# pred= pred.reshape(250,250,3)

# print(np.mean(pred))
# x = []
# for row in pred:
# 	for indx, pixel in enumerate(row):
# 			x = [np.uint8(pixel[0]),np.uint8(pixel[1]),np.uint8(pixel[2])]
# 			x = np.array(x)
# 			row[indx] = x

# pred = pred.astype(int)
# print(np.mean(pred))

# io.imsave("training_image.tif",train_validate[0].reshape(250,250,3))
# io.imsave("output.tif",pred)