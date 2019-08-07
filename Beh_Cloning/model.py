from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from dataset import  Dataset
class Model:
    def __init__(self):
        pass

    def architecture(self, X, y):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=(160, 320, 3)))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Flatten(input_shape=(160, 320, 3)))
        model.add(Dense(20))
        model.add((Dense(1)))
        model.compile(loss='mse', optimizer='adam')
        model.fit(X, y, validation_split=0.2, shuffle=True, epochs=5)
        model.save('model.h5')




d = Dataset()
X, y = d.get_X_y()

m = Model()
m.architecture(X, y)