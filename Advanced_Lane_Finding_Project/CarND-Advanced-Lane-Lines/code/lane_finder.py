import numpy as np
import cv2
import matplotlib.pyplot as plt
from helper import Helper
from thresholder import Thresholder
from transformer import Transformer

class LaneFinder:
    def __init__(self):
        self.left_fit = None
        self.right_fit = None
    
    def hist(self, img):
        # TO-DO: Grab only the bottom half of the image
        # Lane lines are likely to be mostly vertical nearest to the car
        bottom_half = img[img.shape[0]//2:,:]

        # TO-DO: Sum across image pixels vertically - make sure to set `axis`
        # i.e. the highest areas of vertical lines should be larger values
        histogram = np.sum(bottom_half, axis=0)

        return histogram
    
    def find_lane_pixels(self, binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 9
        # Set the width of the windows +/- margin
        margin = 200
        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        leftx = []
        lefty = []
        rightx = []
        righty = []    

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            ### TO-DO: Find the four below boundaries of the window ###
            win_xleft_low = leftx_current - int((margin/2))
            win_xleft_high = leftx_current + int((margin/2))
            win_xright_low = rightx_current - int((margin/2))
            win_xright_high = rightx_current + int((margin/2))

            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),
            (win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),
            (win_xright_high,win_y_high),(0,255,0), 2) 

            ### TO-DO: Identify the nonzero pixels in x and y within the window ###
            good_left_inds = []
            good_right_inds = []


            for x, y in zip(nonzerox, nonzeroy):
                if x >= win_xleft_low and x < win_xleft_high and y >= win_y_low and y < win_y_high:
                    good_left_inds.append((x, y))
                    leftx.append(x)
                    lefty.append(y)

                if x >= win_xright_low and x < win_xright_high and y >= win_y_low and y < win_y_high:
                    good_right_inds.append((x, y))
                    rightx.append(x)
                    righty.append(y)


            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)


            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean([x for x, y in good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean([x for x, y in good_right_inds]))


        return leftx, lefty, rightx, righty, out_img    
    
    def fit_polynomial(self, binary_warped):
        # Find our lane pixels first
        leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(binary_warped)

        ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        try:
            left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
            right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1*ploty**2 + 1*ploty
            right_fitx = 1*ploty**2 + 1*ploty

        ## Visualization ##
        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        # Plots the left and right polynomials on the lane lines
        #plt.plot(left_fitx, ploty, color='yellow')
        #plt.plot(right_fitx, ploty, color='yellow')

        return out_img, {'left_fit' : self.left_fit, 'right_fit' : self.right_fit}
    
    def search_around_poly(self, binary_warped, left_fit, right_fit):

        # HYPERPARAMETER
        # Choose the width of the margin around the previous polynomial to search
        # The quiz grader expects 100 here, but feel free to tune on your own!
        margin = 25

        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        ### TO-DO: Set the area of search based on activated x-values ###
        ### within the +/- margin of our polynomial function ###
        ### Hint: consider the window areas for the similarly named variables ###
        ### in the previous quiz, but change the windows to our new search area ###

        left_poly_x = left_fit[0]*nonzeroy**2 + \
                        left_fit[1]*nonzeroy**1 + \
                        left_fit[2]

        right_poly_x =  right_fit[0]*nonzeroy**2 + \
                        right_fit[1]*nonzeroy**1 + \
                        right_fit[2]

        left_lane_inds = []
        right_lane_inds = []

        for i, x in enumerate(nonzerox):        
            if x <= left_poly_x[i] + margin and x >= left_poly_x[i] - margin:
                left_lane_inds.append(i)

            if x <= right_poly_x[i] + margin and x >= right_poly_x[i] - margin:
                right_lane_inds.append(i)        

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit new polynomials
        left_fitx, right_fitx, ploty = self.fit_poly_2(binary_warped.shape, leftx, lefty, rightx, righty)

        ## Visualization ##
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                  ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                  ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        # Plot the polynomial lines onto the image
        #plt.plot(left_fitx, ploty, color='yellow')
        #plt.plot(right_fitx, ploty, color='yellow')
        ## End visualization steps ##

        return result, {'left_fit' : self.left_fit, 'right_fit' : self.right_fit}    
    
    def fit_poly_2(self, img_shape, leftx, lefty, rightx, righty):
        ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)
        
        # Generate x and y values for plotting
        ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
        ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
        left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
        right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]    

        return left_fitx, right_fitx, ploty

    def overlay_lane(self, img, left_fit, right_fit, tr):
    
        color_warp = np.zeros_like(img).astype(np.uint8)


        ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty[::-1]**2 + right_fit[1]*ploty[::-1] + right_fit[2]

        pts_left = np.array([np.vstack([left_fitx, ploty]).T])
        pts_right = np.array([np.vstack([right_fitx, ploty[::-1]]).T])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, tr.Minv, (img.shape[1], img.shape[0]))
        
        # Combine the result with the original image
        result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

        return result


'''
t = Thresholder()
h = Helper()
img = h.load_image('../test_images/straight_lines1.jpg')  

ksize=3
s_thres = t.hls_threshold(img, thresh=(90, 255))
gradx = t.absolute_threshold(img, orient='x', thresh_min=20, thresh_max=100, sobel_kernel=ksize)
grady = t.absolute_threshold(img, orient='y', thresh_min=20, thresh_max=100, sobel_kernel=ksize)
mag_binary = t.magnitude_threshold(img, sobel_kernel=ksize, mag_thresh=(20, 100))
dir_binary = t.direction_threshold(img, sobel_kernel=ksize, thresh=(0.7, 1.3))

op = np.zeros_like(dir_binary)
op[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (s_thres == 1)] = 1
h.parallel_plots(img,'original', op, 'combined', None, 'gray')

t = Transformer()
corners = [(525, 499), (760, 499), (1026, 673), (270, 673)]
top_down, perspective_M = t.transform(op,corners , offset=100)
h.parallel_plots(op, 'Original', top_down, 'Transformed')

l = LaneFinder()
histogram = l.hist(top_down)
plt.plot(histogram)
plt.show()
histogram.shape

img, fits = l.fit_polynomial(top_down)
plt.imshow(img)
plt.show()

plt.imshow(l.search_around_poly(top_down, fits['left_fit'], fits['right_fit']))
plt.show()
'''