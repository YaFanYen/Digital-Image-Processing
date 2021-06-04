# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 16:48:56 2020

@author: milk
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2

img = cv2.imread('C:/Users/milk3/Documents/Digital Image Processing/HW2/Bird_2.tif',0)
rows, cols = img.shape

#magnitude in log scale
dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = np.log(1 + cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum')

#LPF
crow, ccol = rows/2 , cols/2
mask = np.zeros((rows, cols, 2), np.uint8)
mask[int(crow-30):int(crow+30), int(ccol-30):int(ccol+30), :] = 1

fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

plt.imshow(img_back, cmap = 'gray')
plt.title('Image after LPF')

#HPF
mask = np.ones((rows, cols, 2), np.uint8)
mask[int(crow-30):int(crow+30), int(ccol-30):int(ccol+30), :] = 0

fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

plt.imshow(img_back, cmap = 'gray')
plt.title('Image after HPF')

#Table of top 25 DFT frequencies
img_left = img[0:512, 0:256]
dft_left = cv2.dft(np.float32(img_left),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift_left = np.fft.fftshift(dft_left)
freq = np.fft.fftfreq(np.size(dft_shift_left))
max_number = []
for i in range(25):
    max_number.append(max(freq))
print(max_number)
