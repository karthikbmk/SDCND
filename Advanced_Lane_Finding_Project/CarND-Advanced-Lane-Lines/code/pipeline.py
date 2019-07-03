import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

from helper import Helper
from thresholder import Thresholder
from calibrator import Calibrator
from transformer import Transformer
from lane_finder import LaneFinder

class PipeLine:
    def __init__(self, debug=False):
        
        self.calib_images_path = '../camera_cal/*.jpg'
        self.test_image_path = '../camera_cal/calibration1.jpg'
        self.coeffs_path = '../data/calib_coeffs.pickle'
        
        self.cal = self.calibrate_pipeline()
        
        self.h = Helper()  

        
        self.th = Thresholder()
        self.tr = Transformer()
        self.lane_finder = LaneFinder()        
        
        self.debug = debug
        
        
    def calibrate_pipeline(self):
        cal = Calibrator(x_corners=9, y_corners=6)

        cal.compute_img_pts(self.calib_images_path, False)
    
        cal.calibrate_camera(self.test_image_path)

        cal.store_calib_coeffs(self.coeffs_path)

        return cal
    
    def extract_lanes(self, img):

        self.original_img = img
        
        #Calibrate and Undistort
        self.und_img = self.cal.undistort_image(self.original_img)
        
        if self.debug:
            self.h.parallel_plots(self.original_img, 'Old img', self.und_img, 'Undistorted img') 
        
        
        #HYPER_PARAMS for Thresholding
        ksize=3
        s_thres = self.th.hls_threshold(self.und_img, thresh=(90, 255))
        gradx = self.th.absolute_threshold(self.und_img, orient='x', thresh_min=20, thresh_max=100, sobel_kernel=ksize)
        mag_binary = self.th.magnitude_threshold(self.und_img, sobel_kernel=ksize, mag_thresh=(20, 100))
        dir_binary = self.th.direction_threshold(self.und_img, sobel_kernel=ksize, thresh=(0.7, 1.3))

        self.bin_img = np.zeros_like(dir_binary)
        self.bin_img[((gradx == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (s_thres == 1)] = 1
        
        if self.debug:
            self.h.parallel_plots(self.original_img,'original', self.bin_img, 'combined', None, 'gray')
        
        
        #HYPER_PARAMS for Transforming        
        corners = [(525, 499), (760, 499), (1047, 684), (253, 684)]        
        self.bin_warped = self.tr.transform(self.bin_img,corners , offset=200)

        if self.debug:
            self.h.parallel_plots(self.original_img, 'Original', self.bin_warped, 'Transformed')
        
        histogram = self.lane_finder.hist(self.bin_warped)
        if self.debug:
            plt.plot(histogram)     
            plt.show()
    
        if self.lane_finder.left_fit is None or self.lane_finder.right_fit is None:
            self.poly_fit_img, _ = self.lane_finder.fit_polynomial(self.bin_warped)
        else:
            self.poly_fit_img, _ = self.lane_finder.search_around_poly(self.bin_warped, \
                                                                       self.lane_finder.left_fit, \
                                                                       self.lane_finder.right_fit)
            
        if self.debug:
            plt.imshow(self.poly_fit_img)
        
        self.result = self.lane_finder.overlay_lane(self.original_img, self.lane_finder.left_fit, self.lane_finder.right_fit, self.tr)
        if self.debug:
            plt.imshow(self.result)
        
        return self.result
    
    
p = PipeLine(debug=False)

root_dir = '../'
fname = 'harder_challenge_video.mp4'
clip = VideoFileClip(root_dir + fname)
#clip1 = clip1.subclip(40, 42)
video_clip = clip.fl_image(p.extract_lanes)
video_clip.write_videofile('out_' + fname, audio=False)    
