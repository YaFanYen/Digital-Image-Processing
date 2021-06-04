# Digital-Image-Processing
## DIP_1

Apply intensity transformation function  s=T(r)=H{arctan[(r-128)/32]} to the image.

Plot the figures of the original and output histograms.


## ## DIP_2

Re-synthesize the images using the DFT coefficients (1) inside (r < 30), and (2) outside (r >= 30) the circular region with radius=30 pixels (based on the original image size)


## DIP_3

Consider the image, degraded by mild atmospheric turbulence blurring.

Estimate the parameter k of the model developed by Hufnagel & Stanley.
Construct and plot the restored image using the H(u,v) obtained.


## DIP_4

Consider the RGB color image.

Determine and plot the R, G, B, H, S and I component images.

Sharpen the image by RGB-based and HSI-based schemes.


## DIP_5

Consider the gray-scale image.

Apply the Marr-Hildreth edge detection algorithm to obtain the edge image. 
Plot all the images generated during the entire step-by-step procedure of applying the algorithm. 
Assume two thresholds: 0% and 4% the maximum gray level of the LoG image.

Edge linking by Hough transform: based on the edge map obtained using 4% max{Log} as the threshold, use the Hough transform to perform edge linking.


## DIP_6

Apply Otsu’s optimal global thresholding (single threshold) to R component.

Apply K-means clustering using T = 1, 5 and 10 to the full-color image (RGB) to extract the plums (dividing into 2 clusters). 
