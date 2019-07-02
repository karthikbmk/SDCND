import cv2
import numpy as np
from helper import Helper

class Thresholder:
    def __init__(self):
        pass

    def absolute_threshold(self, img, orient='x', thresh_min=0, thresh_max=255, sobel_kernel=3):

            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Apply x or y gradient with the OpenCV Sobel() function
        # and take the absolute value
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # Create a copy and apply the threshold
        binary_output = np.zeros_like(scaled_sobel)
        # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
        binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
        # Return the result
        return binary_output
    
    def magnitude_threshold(self, img, sobel_kernel=3, mag_thresh=(0, 255)):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag)/255 
        gradmag = (gradmag/scale_factor).astype(np.uint8) 
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

        # Return the binary image
        return binary_output
    
    def direction_threshold(self, img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
        # Apply the following steps to img
        # 1) Convert to grayscale
        # 2) Take the gradient in x and y separately
        # 3) Take the absolute value of the x and y gradients
        # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
        # 5) Create a binary mask where direction thresholds are met
        # 6) Return this mask as your binary_output image
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
        abs_sobelx  = np.sqrt(np.square(sobelx)) 
        abs_sobely  = np.sqrt(np.square(sobely))
        direction = np.arctan2(abs_sobely, abs_sobelx)
        sbinary = np.zeros_like(direction)
        sbinary[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
        return sbinary

    def hls_threshold(self, img, thresh=(0, 255)):
        # 1) Convert to HLS color space
        # 2) Apply a threshold to the S channel
        # 3) Return a binary image of threshold result
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        S = hls[:,:,2]
        binary_output = np.zeros_like(S)
        binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1
        
        return binary_output
    
''' 
h = Helper()
img = h.load_image('../test_images/straight_lines1.jpg')  

t = Thresholder()
op = t.absolute_threshold(img, orient='x', thresh_min=20, thresh_max=100, sobel_kernel=3)
# Plot the result
h.parallel_plots(img,'original', op, 'abs thresholded', None, 'gray')


op = t.magnitude_threshold(img, sobel_kernel=3, mag_thresh=(30, 100))
# Plot the result
h.parallel_plots(img,'original', op, 'mag thresholded', None, 'gray')

op = t.direction_threshold(img, sobel_kernel=15, thresh=(0.7, 1.3))
h.parallel_plots(img,'original', op, 'dir thresholded', None, 'gray')


ksize=3
s_thres = t.hls_threshold(img, thresh=(90, 255))
gradx = t.absolute_threshold(img, orient='x', thresh_min=20, thresh_max=100, sobel_kernel=ksize)
grady = t.absolute_threshold(img, orient='y', thresh_min=20, thresh_max=100, sobel_kernel=ksize)
mag_binary = t.magnitude_threshold(img, sobel_kernel=ksize, mag_thresh=(20, 100))
dir_binary = t.direction_threshold(img, sobel_kernel=ksize, thresh=(0.7, 1.3))

op = np.zeros_like(dir_binary)
op[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (s_thres == 1)] = 1
h.parallel_plots(img,'original', op, 'combined', None, 'gray')


op = t.hls_threshold(img, thresh=(90, 255))
h.parallel_plots(img,'original', op, 'hls thresh', None, 'gray')
'''