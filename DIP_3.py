# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 21:45:11 2020

@author: milk
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
#1
img = cv2.imread('C:/Users/milk3/Documents/Digital Image Processing/HW3/Bird 2 degraded.tif',0)
dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = np.log(1 + cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Fouruer Magnitude Spectrum')
#2
img_org = cv2.imread('C:/Users/milk3/Documents/Digital Image Processing/HW2/Bird_2.tif',0)
img_org = cv2.resize(img_org,(600,600),interpolation = cv2.INTER_CUBIC)
dft_org = cv2.dft(np.float32(img_org),flags = cv2.DFT_COMPLEX_OUTPUT)
H = dft / dft_org
dft_shift_h = np.fft.fftshift(H)
magnitude_spectrum_h = np.log(1 + cv2.magnitude(dft_shift_h[:,:,0],dft_shift_h[:,:,1]))
plt.imshow(magnitude_spectrum_h, cmap = 'gray')
plt.title('Fouruer Magnitude Spectrum of H')
#3
rows, cols = img.shape
crow, ccol = rows/2, cols/2

img_restore = dft / H
img_restore = np.nan_to_num(img_restore)
img_restore_shift = np.fft.fftshift(img_restore)
mask1 = np.zeros((rows, cols, 2), np.uint8)
mask1[int(crow-50):int(crow+50), int(ccol-50):int(ccol+50), :] = 1
fshift1 = img_restore_shift*mask1
f_ishift1 = np.fft.ifftshift(fshift1)
img_back1 = cv2.idft(f_ishift1)
img_back1 = cv2.magnitude(img_back1[:,:,0],img_back1[:,:,1])
plt.imshow(img_back1, cmap = 'gray')
plt.title('Restore Image(radii 50)')
plt.show()

img_restore = dft / H
img_restore = np.nan_to_num(img_restore)
img_restore_shift = np.fft.fftshift(img_restore)
mask2 = np.zeros((rows, cols, 2), np.uint8)
mask2[int(crow-85):int(crow+85), int(ccol-85):int(ccol+85), :] = 1
fshift2 = img_restore_shift*mask2
f_ishift2 = np.fft.ifftshift(fshift2)
img_back2 = cv2.idft(f_ishift2)
img_back2 = cv2.magnitude(img_back2[:,:,0],img_back2[:,:,1])
plt.imshow(img_back2, cmap = 'gray')
plt.title('Restore Image(radii 85)')
plt.show()

img_restore = dft / H
img_restore = np.nan_to_num(img_restore)
img_restore_shift = np.fft.fftshift(img_restore)
mask3 = np.zeros((rows, cols, 2), np.uint8)
mask3[int(crow-120):int(crow+120), int(ccol-120):int(ccol+120), :] = 1
fshift3 = img_restore_shift*mask3
f_ishift3 = np.fft.ifftshift(fshift3)
img_back3 = cv2.idft(f_ishift3)
img_back3 = cv2.magnitude(img_back3[:,:,0],img_back3[:,:,1])
plt.imshow(img_back3, cmap = 'gray')
plt.title('Restore Image(radii 120)')
plt.show()
#4
H = np.nan_to_num(H)
k = -np.log(H[0,0,0]) / pow((2*pow((1-300),2)),5/6)
print(k)
