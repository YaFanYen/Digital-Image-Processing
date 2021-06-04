# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 16:04:01 2020

@author: milk
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
#1 between-class variance
img1 = cv2.imread('C:/Users/milk3/Documents/Digital Image Processing/HW6/fruit on tree.tif')
img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
R = img[:,:,0]
#Curve of between-class variance
hist, bin_edges = np.histogram(R, 256)
hist = hist/sum(hist)
thres = np.arange(0, 256, 1).tolist()
mean = sum(hist*thres)
var = np.zeros(256)
for i in range(1,256,1):
  c1 = sum(hist[:i]*thres[:i])/sum(hist[:i])
  c2 = sum(hist[i:]*thres[i:])/sum(hist[i:])
  var[i] = (sum(hist[:i])*pow(c1-mean,2))+(sum(hist[i:])*pow(c2-mean,2))
plt.title('Between-Class Variance')
plt.plot(thres,var)
plt.show()
#Image of patterns extracted by Otsu’s algorithm
otsu_threshold,img_result = cv2.threshold(R,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
hist = plt.hist(R.ravel(),256,[0,256])

result = np.zeros((img.shape[0],img.shape[1],3), np.uint8)
for i in range(0,img.shape[0]):
  for j in range(0,img.shape[1]):
    if R[i][j] > otsu_threshold:
      result[i][j] = img[i][j]
    else:
      result[i][j] = [128,128,128]

plt.title('Otsu’s result')
plt.imshow(result)
plt.show()
#K-means clustering
pixel_values = img.reshape((-1, 3))
pixel_values = np.float32(pixel_values)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 8
_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
centers = np.uint8(centers)
labels = labels.flatten()
segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(img.shape)
plt.title('k=8')
plt.imshow(segmented_image)
plt.show()
