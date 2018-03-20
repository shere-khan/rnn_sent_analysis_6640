import os
from keras import callbacks
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential


def train(model, trainingdata, testdata):
    # Model params
    timesteps = 350
    dim = 300
    batch_size = 64
    epochs_number = 40

    # Create RNN model
    model = Sequential()
    model.add(LSTM(200, input_shape=(timesteps, dim), return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss="binary_crossentropy", optimizer='rmsprop', metrics=['accuracy'])

    fname = 'weights/keras-lstm.h5'
    if os.path.isfile(fname):
        model.load_weights(fname)

    cbks = [callbacks.ModelCheckpoint(filepath=fname, monitor='val_loss',
                                      save_best_only=True),
            callbacks.EarlyStopping(monitor='val_loss', patience=3)]

    # get all available data samples from data iterators
    model.fit(trainingdata[0], trainingdata[1], batch_size=batch_size, callbacks=cbks,
              epochs=epochs_number, validation_split=0.2, shuffle=True)
    loss, acc = model.evaluate(testdata[0], testdata[1], batch_size=batch_size)
    print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))
