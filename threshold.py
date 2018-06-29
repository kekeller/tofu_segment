

import matplotlib.pyplot as plt
from skimage import data
from skimage import filters
from skimage import exposure
from skimage import io,img_as_uint, img_as_float, filters
from skimage.color import rgb2hsv
from skimage.filters import threshold_local, threshold_otsu, rank, try_all_threshold
from skimage.morphology import disk

import matplotlib.cm as cm

#image = io.imread('./data/train_images/FPSB0070047_RGB1_20170613_172645__35.jpg')
image = io.imread('./tmp/7_img.jpg')
image = io.imread('/home/kevin/Documents/github/tofu_segment/data_full/validate_images/FPSB0070015_RGB1_20170602_155750__04.jpg')

orig = image
image = filters.gaussian(image, sigma=3)
image_blur = image
image = rgb2hsv(image)
camera = image[:,:,0]
val = filters.threshold_otsu(camera)
hist, bins_center = exposure.histogram(camera)

image1 = io.imread('/home/kevin/Documents/github/tofu_segment/data_full/validate_images/FPSB0070015_RGB1_20170602_155750__04_MIRROR.jpg')

orig = image1
image1 = filters.gaussian(image1, sigma=3)

image1 = rgb2hsv(image1)
camera1 = image1[:,:,0]
val1 = filters.threshold_otsu(camera1)
hist1, bins_center1 = exposure.histogram(camera1)

image2 = io.imread('/home/kevin/Documents/github/tofu_segment/data_full/validate_images/FPSB0070015_RGB1_20170602_155750__04_FLIP.jpg')

image2 = filters.gaussian(image2, sigma=3)

image2 = rgb2hsv(image2)
camera2 = image2[:,:,0]
val2 = filters.threshold_otsu(camera2)
hist2, bins_center2 = exposure.histogram(camera2)

image3 = io.imread('/home/kevin/Documents/github/tofu_segment/data_full/validate_images/FPSB0070015_RGB1_20170602_155750__04_ROTATE90.jpg')

image3 = filters.gaussian(image3, sigma=3)

image3 = rgb2hsv(image3)
camera3 = image3[:,:,0]
val3 = filters.threshold_otsu(camera3)
hist2, bins_center2 = exposure.histogram(camera2)


plt.figure(figsize=(12, 3))

plt.subplot(141)
plt.imshow(camera < val, cmap='gray', interpolation='nearest')
plt.axis('off')

plt.subplot(143)
plt.imshow(camera1 < val1, cmap='gray', interpolation='nearest')
plt.axis('off')

plt.subplot(142)
plt.imshow(camera2 < val2, cmap='gray', interpolation='nearest')
plt.axis('off')

plt.subplot(144)
plt.imshow(camera3 < val3, cmap='gray', interpolation='nearest')
plt.axis('off')


# plt.subplot(151)
# plt.imshow(orig)#, cmap='gray', interpolation='nearest')
# plt.axis('off')
# plt.subplot(152)
# plt.imshow(image, cmap='gray', interpolation='nearest')
# plt.axis('off')
# plt.subplot(153)
# plt.imshow(image[:,:,0], cmap='gray', interpolation='nearest')
# plt.axis('off')
# plt.subplot(154)
# plt.imshow(image[:,:,1], cmap='gray', interpolation='nearest')
# plt.axis('off')
# plt.subplot(155)
# plt.imshow(image[:,:,2], cmap='gray', interpolation='nearest')
# plt.axis('off')



# plt.subplot(131)
# plt.imshow(orig)#, cmap='gray', interpolation='nearest')
# plt.axis('off')
# plt.subplot(132)
# plt.imshow(image_blur, cmap='gray', interpolation='nearest')
# plt.axis('off')
# plt.subplot(133)
# plt.imshow(image, cmap='gray', interpolation='nearest')
# plt.axis('off')

# plt.subplot(131)
# plt.imshow(image[:,:,0], cmap='gray', interpolation='nearest')
# plt.axis('off')
# plt.subplot(132)
# plt.imshow(image[:,:,1], cmap='gray', interpolation='nearest')
# plt.axis('off')
# plt.subplot(133)
# plt.imshow(image[:,:,2], cmap='gray', interpolation='nearest')
# plt.axis('off')

# plt.subplot(121)
# plt.imshow(camera < val, cmap='gray', interpolation='nearest')
# plt.axis('off')
# plt.subplot(122)
# plt.plot(bins_center, hist, lw=2)
# plt.axvline(val, color='k', ls='--')




plt.tight_layout()
plt.show()

