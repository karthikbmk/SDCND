from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, Cropping2D
from dataset import  Dataset
class Model:
    def __init__(self):
        pass

    def architecture(self, X, y):
        model = Sequential()
        model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
        model.add(Conv2D(32, (5, 5), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
        model.add(Dropout(rate=0.1))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(20))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')
        model.fit(X, y, validation_split=0.2, shuffle=True, epochs=3)
        model.save('model.h5')




d = Dataset()
X, y = d.get_X_y()

m = Model()
m.architecture(X, y)