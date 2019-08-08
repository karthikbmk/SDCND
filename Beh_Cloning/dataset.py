from helper import Helper
import numpy as np
import cv2
from sklearn.utils import  shuffle
from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self, path = '/Users/karthik/Desktop/beta_simulator_mac/Data/driving_log.csv'):
        self.path = path
        self.root_path = self.path.split('/')
        self.helper = Helper()
        self.augmentors = {'flip'}
        self.aug_factor = len([self.augmentors])

    def get_generator(self, samples, batch_size=32, gen_type= 'train_gen'):
        '''
        :param samples: A list of dictionaries containing the csv dataset.
        :param batch_size: Size of the batch
        :return: A generator, that yeilds a batch of size batch_size * aug_factor from samples.
        '''
        num_samples = len(samples)
        while 1:  # Loop forever so the generator never terminates
            shuffle(samples)
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset + batch_size]

                images = []
                angles = []
                for batch_sample in batch_samples:
                    center_img_name = self.root_path + batch_sample['center'].split('/')[-1]
                    center_img = cv2.imread(center_img_name)
                    center_img = cv2.cvtColor(center_img, cv2.COLOR_BGR2RGB)

                    center_angle = float(batch_sample['angle'])
                    images.append(center_img)
                    angles.append(center_angle)

                    if gen_type == 'train_gen':
                        if 'flip' in self.augmentors:
                            flipped_center_img = self.flipper(center_img)
                            flipped_center_angle = -center_angle

                            images.append(flipped_center_img)
                            angles.append(flipped_center_angle)
                            
                # trim image to only see section with road
                X_train = np.array(images)
                y_train = np.array(angles)

                yield shuffle(X_train, y_train)

    def flipper(self, img):
        rev_img = cv2.flip(img, 1)
        return  rev_img


'''
dataset = Dataset()
X, y =dataset.get_X_y()
'''