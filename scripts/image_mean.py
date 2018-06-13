import glob
from skimage import io,img_as_uint, img_as_float
import numpy as np

from scipy.misc import imread,imsave

# return a list of images 
def read_data(imgDir):
	images = glob.glob(imgDir + "*.jpg")

	train = []

	for im in images:
		n = im.replace(imgDir,'')
		n = n.replace('.jpg','')
		train.append(n)

	x = []
	for i in range(len(train)):
		name = imgDir + str(train[i]) + '.jpg'
		x.append( (io.imread(name) ) )

	return x


imgDir = './data/all_images/'

x = read_data(imgDir)

R = 0
G = 0
B = 0

for img in x:
	R = R + np.mean(img[:,:,0])
	G = G + np.mean(img[:,:,1])
	B = B + np.mean(img[:,:,2])

print("BGR color space")
print(B/len(x))
print(G/len(x))
print(R/len(x))

img = np.zeros([50,50,3],dtype=np.uint8)
img[:,:,0].fill(255) # or img[:] = 255
img[:,:,2].fill(255) # or img[:] = 255

img1 = np.zeros([50,50,3],dtype=np.uint8)
img1[:,:,1].fill(255) # or img[:] = 255

x = []
x.append(img)
x.append(img1)

R = 0
G = 0
B = 0

for image in x:
	R = R + np.mean(image[:,:,0])
	G = G + np.mean(image[:,:,1])
	B = B + np.mean(image[:,:,2])

print(len(x))

print("BGR color space")
print(B/len(x))
print(G/len(x))
print(R/len(x))


img = np.zeros([50,50,3],dtype=np.uint8)
img[:,:,0].fill(67) # or img[:] = 255
img[:,:,1].fill(63) # or img[:] = 255
img[:,:,2].fill(46) # or img[:] = 255

imsave('test.png',img)