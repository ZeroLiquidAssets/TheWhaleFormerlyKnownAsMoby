import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os.path
import numpy as np
import time
from multiprocessing.pool import ThreadPool as Pool
#import tensorflow as tf

def edgedetect(image):
	img = cv.imread('train_small/'+image)
	orig = cv.imread('train_small/'+image)

	"""  #attempt to remove glare from water
	print(np.shape(img))
	
	rows = img.shape[0]
	cols = img.shape[1]
	hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV);
	
	for i in range(0, cols):
		for j in range(0, rows):
			hsv[j, i][1] = 255;
	
	frame = cv.cvtColor(hsv, cv.COLOR_HSV2BGR);
	
	gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
	"""

	#img=cv.GaussianBlur(img,(5,5),0)  #attempt to blur waves

	gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
	ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
	#ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_TRIANGLE)  #sometimes better than otsu

	# noise removal
	kernel = np.ones((2,2),np.uint8)
	#opening = cv.morphologyEx(thresh,cv.MORPH_GRADIENT,kernel, iterations = 4)  #better for poorly defined photos
	opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 4)  #better for well defined photos

	# sure background area
	sure_bg = cv.dilate(opening,kernel,iterations=3)

	# Finding sure foreground area
	dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
	ret, sure_fg = cv.threshold(dist_transform,0.65*dist_transform.max(),255,0)

	# Finding unknown region
	sure_fg = np.uint8(sure_fg)
	unknown = cv.subtract(sure_bg,sure_fg)
	
	# Marker labelling
	ret, markers = cv.connectedComponents(sure_fg)
	# Add one to all labels so that sure background is not 0, but 1
	markers	 = markers+1
	# Now, mark the region of unknown with zero
	markers[unknown==255] = 0
	
	markers = cv.watershed(img,markers)
	img[markers == -1] = [255,0,0]
	markers[markers > 0] = 255
	markers[markers <= 0] = 0

	# attempt to find Contours (doesn't play well with subplots: may be better than markers)
	contours = cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)[-2]
	largest_area=0
	i=-1
	for contour in contours:
		cv.drawContours(orig, contour, -1, (0, 255, 0), 3)
	#cv.drawContours(orig, contours[i], -1, (0, 255, 0), 3)
	#plt.figure()
	#plt.imshow(orig)
	#plt.show()
	
        
	## Find outer contours 
	_, cnts= cv.findContours(thresh,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
	## Draw 
	canvas = np.zeros_like(img)
	cv.drawContours(canvas , contours, -1, (0, 255, 0), 1)
	plt.imshow(canvas)
	plt.show()
	return contours
	""" 


	plt.subplot(221),plt.imshow(orig,cmap = 'gray')
	plt.title('Original Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(222),plt.imshow(img,cmap = 'gray')
	plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(223),plt.imshow(thresh,cmap = 'gray')
	plt.title('Thresh Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(224),plt.imshow(markers,cmap = 'gray')
	plt.title('Marker Image'), plt.xticks([]), plt.yticks([])
	plt.show()
	#return markers
	"""
names=[]
outlines=[]
for filename in os.listdir('train_small/'):
	if filename.endswith(".jpg"):
		names.append(filename)

pool=Pool(1)  #Number of CPU cores
start_time = time.time()
#for item in names:
#	edgedetect(item)
if __name__=='__main__':
        outlines.append(pool.map(edgedetect,names))


print("--- %s seconds ---" % (time.time() - start_time))

