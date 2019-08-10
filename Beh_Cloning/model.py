from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, Cropping2D, Lambda
from dataset import  Dataset
from keras.callbacks import ModelCheckpoint
from math import ceil
from helper import Helper
import matplotlib
matplotlib.use('PS')
import  matplotlib.pyplot as plt
import json

class Model:
    def __init__(self):
        self.model  = self.architecture()
        self.help = Helper()
        self.params = self.help.load_json('./misc/params.json')

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
        model.add(Conv2D(256, (3, 3), padding="valid", activation="relu"))
        model.add(Dropout(rate=0.1))
        model.add(Conv2D(512, (3, 3), padding="valid", activation="relu"))
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

    def store_history(self, history_object):
        hist_path = self.params['history_path']
        plt.plot(history_object.history['loss'])
        plt.plot(history_object.history['val_loss'])
        plt.title('model mean squared error loss')
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validation set'], loc='upper right')
        plt.savefig(hist_path)

    def freeze_layers(self, model, freeze_layer_ids):
        '''
        :param model:
        :param freeze_layer_ids: Layer ids to be frozen
        :return: Frozen model
        '''
        for layer_id in freeze_layer_ids:
            model.layers[layer_id].trainable = False
        return model

    def fit_model(self, train_type, model, batch_size=32, out_model_name='model' ,epochs=4):

        #Load dataset
        csv_path = self.params[train_type]
        d = Dataset(csv_path)
        train_gen, val_gen = d.get_train_val_gens(batch_size= batch_size)

        #Fit Model
        steps_per_ep = ceil((d.train_size * d.aug_factor) / batch_size)
        val_steps = ceil((d.val_size * d.aug_factor) / batch_size)

        models_path = self.params['models_path']
        filepath = models_path + out_model_name + "-{epoch:02d}-{val_loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

        history_object = model.fit_generator(train_gen, steps_per_epoch=steps_per_ep, validation_data=val_gen, \
                                 validation_steps=val_steps, epochs=epochs, verbose=1, callbacks=[checkpoint])

        #Store history
        self.store_history(history_object)


    def transfer_learn(self, model, train_type, freeze_layer_ids, batch_size=32, out_model_name = 'tl_model'):

        #Freeze Layers
        frozen_model = self.freeze_layers(model, freeze_layer_ids)

        #Define loss fn and optimizer
        model.compile(loss='mse', optimizer='adam')

        #Fit new model
        self.fit_model(train_type, frozen_model, batch_size, out_model_name, epochs=3)

    def do_transfer_learn(self, train_type, old_model_name, trained_version):

        #Load old model
        old_model = load_model(self.params['models_path'] + old_model_name)
        print ('loaded old model')

        freeze_layer_ids = list(range(10))
        batch_size = 16
        out_model_name = 'tl_v' + str(trained_version + 1) +'_model'

        self.transfer_learn(old_model, train_type, freeze_layer_ids, batch_size, out_model_name)
        print ('transfer learning complete.')

'''
Fit Model code
m = Model()
m.fit_model(model=m.model, batch_size=32, out_model_name='dummy_model')
'''

'''
Transfer learning code
'''
m = Model()
trained_version = 11
m.do_transfer_learn('tf_csv_path', 'tl_v' + str(trained_version) + '_model-03-0.03.hdf5', trained_version)


