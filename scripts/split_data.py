import glob, random, shutil, os

imgDir = './data/train_images/'
maskDir = './data/train_masks/'

valImgDir = './data/validate_images/'
valMaskDir = './data/validate_masks/'

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
	x.append( name)

	name = maskDir + str(train[i]) + '.png'
	y.append( name )

print(len(x))

valImg = random.sample(x,100)

for img in valImg:
	nameImg = img
	nameMask = img.replace(imgDir,maskDir)
	nameMask = nameMask.replace('.tif','.png')

	nameImgVal = img.replace(imgDir,valImgDir)
	nameMaskVal = nameMask.replace(maskDir,valMaskDir)

	shutil.move(nameImg,nameImgVal)

	shutil.move(nameMask,nameMaskVal)


