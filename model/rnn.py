import sys
from keras import callbacks
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
from gensim.models import Word2Vec


def train(model):
    timesteps = 350
    dim = model.layer1_size
    batch_size = 64
    epo = 40

    model = Sequential()
    model.add(LSTM(200, input_shape=(timesteps, dim), return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss="binary_crossentropy", optimizer='rmsprop', metrics=['accuracy'])
    fname = 'weights/keras-lstm.h5'
    model.load_weights(fname)
    cbks = [callbacks.ModelCheckpoint(filepath=fname, monitor='val_loss',
                                      save_best_only=True), callbacks.EarlyStopping(
        monitor='val_loss', patience=3)]

    # train_iterator = DataIterator

if __name__ == '__main__':
    model = Word2Vec.load("../preprocess/word2vecmodel")
    # train(model)
    # print('length of vocab', len(model.wv.vocab))
    # print(model.layer1_size)
