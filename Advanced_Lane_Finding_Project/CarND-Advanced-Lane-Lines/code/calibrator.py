import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
from helper import Helper

class Calibrator:
    def __init__(self, x_corners, y_corners):        

        self.nx = x_corners
        self.ny = y_corners        
        
        #Has to be computed in compute_img_pts()
        self.objpoints = []
        self.imgpoints = []        
        
        #Calibration params. should be computed in calibrate_camera()
        self.calib_params = {}
    def compute_img_pts(self, calib_images_path, show_corners=True):
        
        objp = np.zeros((self.ny * self.nx, 3), np.float32)
        objp[:,:2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1, 2)
            
        # Make a list of calibration images
        images = glob.glob(calib_images_path)

        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)

            
            # If found, add object points, image points
            if ret == True:
                self.objpoints.append(objp)
                self.imgpoints.append(corners)

                # Draw and display the corners
                cv2.drawChessboardCorners(img, (self.nx, self.ny), corners, ret)        
                
                if show_corners:
                    plt.imshow(img)
                    plt.show()
        
        print ('corners found. please check self.imgpoints')
    
    def calibrate_camera(self, image_path):
        if len(self.imgpoints) == 0 or len(self.objpoints) == 0:
            raise Exception("calibration unsuccessful. Have you executed compute_img_pts() ?")
        
        img = cv2.imread(image_path)
        img_size = (img.shape[1], img.shape[0])
        
        self.calib_params['ret'], \
        self.calib_params['mtx'], \
        self.calib_params['dist'], \
        self.calib_params['rvecs'], \
        self.calib_params['tvecs'] = cv2.calibrateCamera(self.objpoints, self.imgpoints, img_size, None, None)
        
        return self.calib_params
        
        
    def store_calib_coeffs(self, store_path):
        if len(self.calib_params) == 0:
            raise Exception('calib_params is empty. Have you run calibrate_camera ??')
            
        pickle.dump( self.calib_params, open( store_path, "wb" ) )
        print ('stored calib coeffs @ ' + store_path)
        
    
    def undistort_image(self, img):
        dst = cv2.undistort(img, self.calib_params['mtx'], self.calib_params['dist'], None, self.calib_params['mtx'])
        return dst


'''
h = Helper()

cal = Calibrator(x_corners=9, y_corners=6)

calib_images_path = '../camera_cal/*.jpg'
cal.compute_img_pts(calib_images_path, False)

test_image_path = '../camera_cal/calibration1.jpg'
cal.calibrate_camera(test_image_path)

cal.store_calib_coeffs('../data/calib_coeffs.pickle')

img = h.load_image(test_image_path)
und_img = cal.undistort_image(img)
h.parallel_plots(img, 'Old img', und_img, 'Undistorted img')
'''