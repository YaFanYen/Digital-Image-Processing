# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 13:17:25 2020

@author: milk
"""
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
#result of LoG
img = cv2.imread('C:/Users/milk3/Documents/Digital Image Processing/HW5/Car On Mountain Road.tif',0)
blur = cv2.GaussianBlur(img,(3,3),0)
laplacian = cv2.Laplacian(blur,cv2.CV_64F)
laplacian1 = laplacian/laplacian.max()
plt.imshow(laplacian1, cmap = 'gray')
#zero-crossing
def Zero_crossing(image):
    z_c_image = np.zeros(image.shape)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            negative_count = 0
            positive_count = 0
            neighbour = [image[i+1, j-1],image[i+1, j],image[i+1, j+1],
                         image[i, j-1],image[i, j+1],image[i-1, j-1],image[i-1, j],image[i-1, j+1]]
            d = max(neighbour)
            e = min(neighbour)
            for h in neighbour:
                if h>0:
                    positive_count += 1
                elif h<0:
                    negative_count += 1
            z_c = ((negative_count > 0) and (positive_count > 0))
            if z_c:
                if image[i,j]>np.max(laplacian1)*0.04:
                    z_c_image[i, j] = image[i,j] + np.abs(e)
                elif image[i,j]<np.max(laplacian1)*0.04:
                    z_c_image[i, j] = np.abs(image[i,j]) + d
    z_c_norm = z_c_image/z_c_image.max()*255
    z_c_image = np.uint8(z_c_norm)
    return z_c_image

zerocrossing = Zero_crossing(laplacian1)
plt.imshow(zerocrossing, cmap = 'gray')
#Hough parameter space
def hough_line(img, angle_step=1, lines_are_white=True, value_threshold=5):
    thetas = np.deg2rad(np.arange(-90.0, 90.0, angle_step))
    width, height = img.shape
    diag_len = int(round(math.sqrt(width * width + height * height)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)
    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint8)
    are_edges = img > value_threshold if lines_are_white else img < value_threshold
    y_idxs, x_idxs = np.nonzero(are_edges)
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for t_idx in range(num_thetas):
            rho = diag_len + int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))
            accumulator[rho, t_idx] += 1
    return accumulator, thetas, rhos
edges = cv2.Canny(img, 50, 200)
accumulator, thetas, rhos = hough_line(zerocrossing)
plt.imshow(accumulator, aspect='auto', cmap='gray', 
           extent=[np.rad2deg(thetas[-1]), np.rad2deg(thetas[0]), rhos[-1], rhos[0]])
#linked edges alone and overlapped on the original image
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=10, maxLineGap=250)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
plt.imshow(img, cmap = 'gray')


