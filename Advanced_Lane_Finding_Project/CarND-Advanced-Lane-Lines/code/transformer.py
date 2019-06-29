import numpy as np
import cv2
from helper import Helper


class Transformer:
    def __init__(self):
        pass
    
    def transform(self, img, corners, offset):

        x = img.shape[1]
        y = img.shape[0]
        
        src = np.float32(corners)

        img_size = (x, y)    

        dst = np.float32([(0 + offset, 0 + offset), \
                          (x - offset, 0 + offset), \
                          (x - offset, y - offset), \
                          (0 + offset, y - offset)])

        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
        return warped, M


'''
h = Helper()
img = h.load_image('../test_images/straight_lines1.jpg')  
t = Transformer()
corners = [(525, 499), (760, 499), (1026, 673), (270, 673)]
top_down, perspective_M = t.transform(img,corners , offset=40)
h.parallel_plots(img, 'Original', top_down, 'Transformed')
'''
