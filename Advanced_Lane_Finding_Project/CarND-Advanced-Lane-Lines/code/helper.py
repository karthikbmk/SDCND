import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
import copy
import os
import tempfile

'''
import matplotlib
matplotlib.use('agg')

plt.ioff()
'''

class Helper:
    
    def __init__(self):
        pass
    
    def parallel_plots(self, img1, title1, img2, title2, cmap1 = None, cmap2 = None):
        ''' Utility function to plot 2 images parallely
        '''
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(img1, cmap=cmap1)
        ax1.set_title(title1, fontsize=50)
        ax2.imshow(img2, cmap = cmap2)
        ax2.set_title(title2, fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()
        
        return
    
    def plot_image(self, img, title):

        plt.imshow(img)
        plt.title(title)
        plt.show()

    def load_image(self, path, bgr2rgb=True):
        img = cv2.imread(path)
        if bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def overlay_text(self, img, text):
               
        #Tweak these params as required.
        x = int(img.shape[1]/5)
        y = int(img.shape[1]/16)
        bottomLeftCornerOfText = (x, y)
        fontScale              = 2
        fontColor              = (255,255,255)
        lineType               = 2
        font                   = cv2.FONT_HERSHEY_SIMPLEX        


        cv2.putText(img, text, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)
        
        return img
    
    def img_to_channel(self, img, channel):

        if channel == 'g':
            img = img[:,:,1]        
            npad = ((0,0), (0,0), (1,1))

        elif channel == 'r':
            img = img[:,:,0]
            npad = ((0,0), (0,0), (0,2))

        elif channel == 'b':
            img = img[:,:,2]
            npad = ((0,0), (0,0), (2,0)) 

        img = np.expand_dims(img, axis=2)
        img = np.pad(img, npad,  'constant', constant_values=(0, 0))

        return img    

    
    def image_merger(self, out_shape, files):

        result = Image.new("RGB", out_shape)
        
        prev = None
        
        lane_x = 0 
        lane_y = 0
        lane_w = int(0.7*out_shape[0])
        lane_h = int(out_shape[1])    
        
        first_debug = True
        
        for file in files:
            
            img = copy.deepcopy(file['img'])
                        
            if len(img.shape) <= 2:
                img = np.stack((img,)*3, axis=-1)                
                img = (img * 255).astype(np.uint8)   
            
            if not isinstance(img, Image.Image) :
                img = Image.fromarray(img.astype(np.uint8))
                                  
            
            if file['type'] == 'lane':

                img = img.resize((lane_w, lane_h))
                result.paste(img, (lane_x, lane_y, lane_x + lane_w, lane_y + lane_h))        
                prev = 'lane'

            else:
                x = lane_w
                if first_debug:
                    y = 0
                else:
                    y = y + h

                w = out_shape[0] - x
                h = int(0.25 * out_shape[1])

                prev = 'debug'
                first_debug = False

                img = img.resize((w, h))
                
                result.paste(img, (x, y, w + x, h + y))

        return np.array(result)    
    

    def hist_to_img(self, histogram):

        fig = plt.figure()
        ax = plt.subplot(111)
        ax.plot(histogram)

        fd, path = tempfile.mkstemp()
        try:
            with os.fdopen(fd, 'w') as tmp:
                # do stuff with temp file
                fname = path + 'hist.png'        
                fig.savefig(fname)

                ax.remove()
                plt.pause(0.05)                
                
                im  = cv2.imread(fname)                                
                return im
        finally:
            os.remove(fname)    