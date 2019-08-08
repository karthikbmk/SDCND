from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, Cropping2D, Lambda
from dataset import  Dataset
from math import ceil
import  matplotlib.pyplot as plt
class Model:
    def __init__(self):
        self.model  = self.architecture()

    def architecture(self):
        '''
        Specify model Architecture here.
        :return: model object
        '''
        model = Sequential()
        model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(160, 320, 3)))
        model.add(Cropping2D(cropping=((70, 20), (0, 0))))
        model.add(Conv2D(32, (5, 5), padding="valid", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), padding="valid", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (3, 3), padding="valid", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Dropout(rate=0.1))
        model.add(Dense(32))
        model.add(Dropout(rate=0.1))
        model.add(Dense(16))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')
        return model

    def store_history(self, history_object, hist_path = './history.png'):
        plt.plot(history_object.history['loss'])
        plt.plot(history_object.history['val_loss'])
        plt.title('model mean squared error loss')
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validation set'], loc='upper right')
        plt.savefig(hist_path)


    def fit_model(self, batch_size=32):
        d = Dataset()
        train_gen, val_gen = d.get_train_val_gens(batch_size= batch_size)

        steps_per_ep = ceil((d.train_size * d.aug_factor) / batch_size)
        val_steps = ceil((d.val_size * d.aug_factor) / batch_size)

        history_object = self.model.fit_generator(train_gen, steps_per_epoch=steps_per_ep, validation_data=val_gen, \
                                 validation_steps=val_steps, epochs=5, verbose=1)

        self.store_history(history_object)
        self.model.save('model.h5')

m = Model()
m.fit_model()




