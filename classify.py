from sklearn.ensemble import RandomForestClassifier
import glob
from skimage import io
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
	return mask

train,label = read_data("./images")
mask = convert_label_mask(label)

train_validate,label_validate = read_data("./predict")
mask_validate = convert_label_mask(label_validate)

print(len(train_validate))

clf = RandomForestClassifier(verbose=1)

clf.fit(train, mask)
pred = clf.predict(train_validate)
pred = np.array(pred)
pred= pred.reshape(250,250,3)

print(np.mean(pred))
for row in pred:
	for indx, pixel in enumerate(row):
		x = []
		if np.mean(pixel) > 0:
			x = [np.uint8(pixel[0]),np.uint8(pixel[1]),np.uint8(pixel[2])]
			x = np.array(x)
			row[indx] = x

io.imsave("out.tif",mask[0].reshape(250,250,3))
io.imsave("out1.tif",train_validate[0].reshape(250,250,3))
io.imsave("out2.tif",pred)