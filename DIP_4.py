# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 12:56:30 2020

@author: milk
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
#1 RGB
img1 = cv2.imread('C:/Users/milk3/Documents/Digital Image Processing/HW4/Bird 3 blurred.tif')
img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
R = img[:,:,0]
G = img[:,:,1]
B = img[:,:,2]
plt.imshow(R, cmap = 'gray')
plt.title('R')
plt.imshow(G, cmap = 'gray')
plt.title('G')
plt.imshow(B, cmap = 'gray')
plt.title('B')
#1 HSI
rows, cols = R.shape
from skimage.color import rgb2hsv, hsv2rgb
img_HSI = rgb2hsv(img)
H = 255 * img_HSI[:,:,0]
S = 255 * img_HSI[:,:,1]
I = 255 * img_HSI[:,:,2]
H = H.astype(np.uint8)
S = S.astype(np.uint8)
I = I.astype(np.uint8)
plt.imshow(H, cmap = 'gray')
plt.title('H')
plt.imshow(S, cmap = 'gray')
plt.title('S')
plt.imshow(I, cmap = 'gray')
plt.title('I')
#2 RGB-based sharpened images
kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
RGB_sharpen = cv2.filter2D(img, -1, kernel)
RGB_sharpen_result = img + RGB_sharpen
plt.imshow(RGB_sharpen_result)
plt.title('RGB-based sharpening')
#2 HSI-based sharpened images
HSI_sharpen = cv2.filter2D(img_HSI, -1, kernel)
HSI_sharpen_result = img_HSI + HSI_sharpen
HSI_sharpen_result2 = hsv2rgb(HSI_sharpen_result)
plt.imshow(HSI_sharpen_result2)
plt.title('HSI-based sharpening')
#2 Difference image
Difference = HSI_sharpen - RGB_sharpen
Difference2 = np.zeros([rows,cols])
for i in range(rows):
    for j in range(cols):
        Difference2[i,j] = pow(pow(Difference[i,j,0], 2) 
        + pow(Difference[i,j,1], 2) + pow(Difference[i,j,2], 2), 1/2)
Difference2 = Difference2.astype(np.uint8)
plt.imshow(Difference2, cmap = 'gray')
plt.title('Difference image')
