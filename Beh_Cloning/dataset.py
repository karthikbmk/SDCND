from helper import Helper
import numpy as np
import cv2

class Dataset:
    def __init__(self, path = '/Users/karthik/Desktop/beta_simulator_mac/Data/driving_log.csv'):
        self.path = path
        self.helper = Helper()

    def get_X_y(self):


        images = []
        steering_angles = []
        lines = self.helper.csv_to_list(self.path)

        for line in lines:
            center_img_path = line['center']
            image = cv2.imread(center_img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)

            steering_angle = float(line['angle'])
            steering_angles.append(steering_angle)

        X = np.array(images)
        y = np.array(steering_angles)


        return X, y

'''
dataset = Dataset()
X, y =dataset.get_X_y()
'''