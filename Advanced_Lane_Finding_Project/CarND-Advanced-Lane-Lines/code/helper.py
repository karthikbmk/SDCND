import matplotlib.pyplot as plt
import cv2

class Helper:
    
    def __init__(self):
        pass
    
    def parallel_plots(self, img1, title1, img2, title2):
        ''' Utility function to plot 2 images parallely
        '''
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(img1)
        ax1.set_title(title1, fontsize=50)
        ax2.imshow(img2)
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
