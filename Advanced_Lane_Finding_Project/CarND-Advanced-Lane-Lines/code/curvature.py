import numpy as np
import math

class Curvature:
    def __init__(self, ym_per_pix, xm_per_pix):
        self.ym_per_pix = ym_per_pix
        self.xm_per_pix = xm_per_pix
    
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