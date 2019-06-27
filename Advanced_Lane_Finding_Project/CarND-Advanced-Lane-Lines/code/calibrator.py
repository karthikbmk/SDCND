import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

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
            print ("calibration unsuccessful. Have you executed compute_img_pts() ?")
            return
        
        img = cv2.imread(image_path)
        img_size = (img.shape[1], img.shape[0])
        
        self.calib_params['ret'], \
        self.calib_params['mtx'], \
        self.calib_params['dist'], \
        self.calib_params['rvecs'], \
        self.calib_params['tvecs'] = cv2.calibrateCamera(self.objpoints, self.imgpoints, img_size, None, None)
        
        return self.calib_params
        
        
    def store_calib_coeffs(self, store_path):
        pickle.dump( self.calib_params, open( store_path, "wb" ) )
        print ('stored calib coeffs @ ' + store_path)
        
    
    def undistort_image(self, image_path):
        img = cv2.imread(image_path)
        dst = cv2.undistort(img, self.calib_params['mtx'], self.calib_params['dist'], None, self.calib_params['mtx'])
        return dst
    
    def plot_image(self, img, title, is_image_path):
        
        if is_image_path:
            img = cv2.imread(img)
        
        plt.imshow(img)
        plt.title(title)
        plt.show()
        
    
