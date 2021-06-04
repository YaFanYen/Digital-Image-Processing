# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 14:51:01 2020

@author: milk
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

#transform
def T(r, r1, s1, r2, s2): 
    if r < r1 :
        return (s1 / r1) * r 
    elif r1 <= r and r < r2 : 
        return ((s2 - s1)/(r2 - r1)) * (r - r1) + s1 
    else: 
        return ((255 - s2)/(255 - r2)) * (r - r2) + s2 

#parameters
r1 = 70
s1 = 30
r2 = 140
s2 = 225

#data
img = cv.imread('C:/Users/milk3/Documents/Digital Image Processing/HW1/Bird feeding 3 low contrast.tif',0)
result = np.vectorize(T)
output = result(img, r1, s1, r2, s2) 
cv.imwrite('contrast_stretch2.jpg', output)

# figure of T(r)
y = []
for x in range(0,256):
    if x < r1 :
        y.append((s1 / r1) * x)
    elif r1 <= x and x < r2 :
        y.append(((s2 - s1)/(r2 - r1)) * (x - r1) + s1)
    else :
        y.append(((255 - s2)/(255 - r2)) * (x - r2) + s2)
x = np.arange(0,256,1)
plt.plot(x,y)
plt.title('s = T(r)')
plt.xlabel('input intensity level (r)')
plt.ylabel('output intensity level (s)')
plt.show()

#histogram
plt.hist(img.flatten(), 256, [0,256])
plt.xlim([0,256])
plt.title('org_histogram') 
plt.show()

plt.hist(output.flatten(), 256, [0,256])
plt.xlim([0,256])
plt.title('output_histogram') 
plt.show()

