## Advance Lane Finder

---
![alt text][video_gif]

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undist.png "Undistorted"
[image2]: ./output_images/undist_road.png "Road Transformed"
[image3]: ./output_images/thresh.png "Binary Example"
[image4]: ./output_images/transformed.png "Warp Example"
[image5]: ./output_images/polyfit.png "Fit Visual"
[image6]: ./output_images/lane_fit.png "Output"
[video1]: ./output/out_project_video.mp4 "Video"
[video_gif]: ./output/out_project_video.gif "Undistorted"


### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Code for camera calibration is present in `code/calibrate.py`

* Compute the object points and store it on objpoints variable
* Compute the image points of all chessboard images imgpoints variable
* Use cv2.calibrateCamera to compute camera matrix (mtx) and distortion coeffs (dist)
* Use cv2.undistort to undistort the image
  

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Code for thresholding is at `code/thresholder.py`  
Used the combination of following techniques with a Sobel Kernel of size 3: 

| THRESHOLDER        | MIN_VAL           | MAX_VAL  | MISC  |
| ------------- |:-------------:| -----:|:-----:|
| HLS      | 200 | 255 | N/A |
| Absolute      | 20      |   100 | orient='x' |
| Magnitude | 20      |    100 | N/A |
| Direction | 0.7      |    1.3 | N/A |


![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Code for transformer is at `code/transformer.py`

```python
src = [(525, 499), (760, 499), (1047, 684), (253, 684)]
dst = np.float32([(0 + offset, 0 + offset), \
                  (x - offset, 0 + offset), \
                  (x - offset, y), \
                  (0 + offset, y)])
```

* offset was set to be 100 and x and y are size of images in x and y axes respectively
* cv2.getPerspectiveTransform and cv2.warpPerspective were used to do the perspective transformation
![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Code for lane fitting is at `code/LaneFinder.find_lane_pixels`
Steps:  
* Construct a histogram of the image to figure the pixel intensities
* Divide the image into 2 halves. The regions of the image having the highest intensities in the left and right halves correspond to left and right lanes
* Use a window approach to identify the mean position for the next lane pixel. Repeat this for all pixels, for left and right lanes
* Once all lane points are found, fit 2nd order polynomials for the left and right lanes.
* 2 polynomials obtained from the above step correspond to left and right lanes
* In order to speed up the computation for lanes, for the next frame, use the previous lane polynomials. Repeat this for all frames

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

* Radius of curavature is calculated in `code/curvature.measure_curvature_real`
```python
    def curve_radius(self, A, y, B):
        dr = math.fabs(2*A)
        nr = (2*A*y + B)**2.0
        nr += 1
        nr = math.pow(nr, (1.5))

        return nr / dr    
    def measure_curvature_real(self, leftx, rightx, ploty):
        '''
        Calculates the curvature of polynomial functions in meters.
        '''
        ym_per_pix = self.ym_per_pix
        xm_per_pix = self.xm_per_pix
        
        left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)        
        
        # Define y-value where we want radius of curvature        
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty) * ym_per_pix

        ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
        left_curverad = self.curve_radius(left_fit_cr[0], y_eval, left_fit_cr[1])
        right_curverad = self.curve_radius(right_fit_cr[0], y_eval, right_fit_cr[1])

        return {'left' : int(left_curverad), 'right' : int(right_curverad)} 
```
* Position of vehicle wrt center is calculated in `code/curvature.compute_offset`
```python
    def compute_offset(self, img_center_x, lane_center_x):
        offset = img_center_x - lane_center_x

        if abs(offset) == offset:
            direction = 'right'
        else:
            direction = 'left'
            
        return round(abs(offset * self.xm_per_pix),2), direction
```
#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Code for lane overlays is at `code/lane_finder.overlay_lane`

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Challenges Faced:  
1. Finding the right set of hyperparams was time consuming 
2. Fixed a bug, where in the right lane's co-ordinates were inverted, because of lanes were wrongly plotted 
3. Had to spend quite some time figuring out how to use OpenCV for overlaying images/text, fill polygons etc  
4. Had to spend time in displaying the debug information, along the plotted lanes in the output video  

Scenarios where pipeline will likely fail:  
1. Might fail on sharp turns, wherein polynomial orders might be greater than 2  
2. Might get confused for potholes such lane lines 
3. Night Light settings wherein lanes aren't visible  

How to make a robust pipeline:
1. Use a Deep Learning based semantic segmentation for more robust detectors 
