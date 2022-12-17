import pickle
import neptune.new as neptune
from tensorflow import keras
from keras.layers import Dense, Activation, Flatten, Dropout
from sklearn.model_selection import train_test_split

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)

X = X/255.0

model = keras.Sequential()

model.add(Flatten(input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(Dense(1024))
model.add(Dropout(.2))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Dropout(.2))
model.add(Activation('relu'))
model.add(Dense(45))
model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'mse'])
# fit model
history = model.fit(X, y, batch_size = 10, epochs=13, verbose=1, validation_split = 0.2)
# evaluate the model
model.save("wh.model")