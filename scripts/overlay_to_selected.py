
import glob, random, shutil, os

imageDir = '/home/kevin/Documents/Data_Set/LabeledData/modified_data/IMAGES/'
maskDir = '/home/kevin/Documents/Data_Set/LabeledData/modified_data/MASK/'

selectMaskDir = '/home/kevin/Documents/Data_Set/LabeledData/modified_data/masks/'
selectImageDir = '/home/kevin/Documents/Data_Set/LabeledData/modified_data/images/'

overlayDir = '/home/kevin/Documents/Data_Set/LabeledData/modified_data/OVERLAY/'

def overlay_to_Mask_Images(overlayDir,maskDir,imageDir,selectMaskDir,selectImageDir):
	"""
	Copy tiff images from original folder and save matching ones for data set
	"""
	overPath = os.path.join(overlayDir) + '*.png'
	print(overPath)

	if not (os.path.exists(selectMaskDir)):
		os.mkdir(selectMaskDir)

	if not (os.path.exists(selectImageDir)):
		os.mkdir(selectImageDir)

	for overlayPath in glob.glob( overPath ):
		#print(overlayPath)
		img = overlayPath.replace('.png','.tif')
		name = img.replace(overlayDir,imageDir)

		name_new = img.replace(overlayDir,selectImageDir)
		shutil.copy(name,name_new)

	for overlayPath in glob.glob( overPath ):
		name = overlayPath.replace(overlayDir,maskDir)
		print(name)
		
		name_new = overlayPath.replace(overlayDir,selectMaskDir)
		shutil.copy(name,name_new)


overlay_to_Mask_Images(overlayDir,maskDir,imageDir,selectMaskDir,selectImageDir)