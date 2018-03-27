import numpy as np
import tensorflow as tf
from preprocess import util
import os, nltk, random, pickle, time
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter


n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 2
batch_size = 100

lemmatizer = WordNetLemmatizer()
hm_lines = 100000000

def tok_and_lem(file, lexicon):
    for f in os.listdir(file):
        with open(file + f, 'r') as fi:
            for l in fi:
                words = word_tokenize(l.lower())
                words = [lemmatizer.lemmatize(i) for i in words]
                lexicon += list(set(words))

def create_lexicon(pos, neg, testpos, testneg):
    files, lexicon = [], []
    start = time.time()

    tok_and_lem(pos, lexicon)
    tok_and_lem(neg, lexicon)
    tok_and_lem(testpos, lexicon)
    tok_and_lem(testneg, lexicon)

    w_counts = Counter(lexicon)

    l2 = []
    for w in w_counts:
        if (250 > w_counts[w] > 1):
            l2.append(w)

    print("Lexicon size = ", len(l2))
    print("Creating lexicon took %.8f seconds" % (time.time() - start))
    return l2

def sample_handling(sample, lexicon, classification):
    featureset = []
    with open(sample, 'r') as f:
        contents = f.readlines()
        # todo change here to do averages
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))

            for word in current_words:
                if (word.lower() in lexicon):
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1

            features = list(features)
            featureset.append([features, classification])

    return featureset

def create_feature_sets_and_labels(pos, neg, testpos, testneg, test_size=0.1):
    start = time.time()
    lexicon = create_lexicon(pos, neg, testpos, testneg)
    features = []
    for f in os.listdir(pos):
        features += sample_handling(pos + f, lexicon, [1, 0])

    for f in os.listdir(neg):
        features += sample_handling(neg + f, lexicon, [0, 1])

    random.shuffle(features)
    features = np.array(features)

    train_x = list(features[:, 0])
    train_y = list(features[:, 1])

    features = []
    for f in os.listdir(testpos):
        features += sample_handling(testpos + f, lexicon, [1, 0])

    for f in os.listdir(testneg):
        features += sample_handling(testneg + f, lexicon, [0, 1])

    random.shuffle(features)
    features = np.array(features)

    test_x = list(features[:, 0])
    test_y = list(features[:, 1])

    print("Creating Sets and Labels took %.8f seconds" % (time.time() - start))
    return train_x, train_y, test_x, test_y

def neural_network_model(data, train_x):
    start = time.time()
    hidden_1_layer = {
        'weights': tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
        'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {
        'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
        'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {
        'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
        'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes])), }

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']
    print("Creating NN took %.8f seconds" % (time.time() - start))

    return output

def train_neural_network(x, train_x, train_y, test_x, test_y, y):
    start2 = time.time()
    prediction = neural_network_model(x, train_x)
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0

            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size

                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size

            print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))
    print("Training NN took %.8f seconds" % (time.time() - start2))

if __name__ == '__main__':
    begin = time.time()
    # pos, neg, testpos, testneg = '/home/john/Desktop/pos/', '/home/john/Desktop/neg/', '/home/john/Desktop/test/pos/', '/home/john/Desktop/test/neg/'
    pos = '/home/justin/pycharmprojects/rnn_sent_analysis_6640/dataset/train/pos/'
    neg = '/home/justin/pycharmprojects/rnn_sent_analysis_6640/dataset/train/neg/'
    testpos = '/home/justin/pycharmprojects/rnn_sent_analysis_6640/dataset/test/pos/'
    testneg = '/home/justin/pycharmprojects/rnn_sent_analysis_6640/dataset/test/neg/'
    train_x, train_y, test_x, test_y = create_feature_sets_and_labels(pos, neg, testpos,
                                                                      testneg)

    # with open('sentiment_set.pickle', 'wb') as f:
    #	pickle.dump([train_x,train_y,test_x,test_y],f)

    x = tf.placeholder('float', [None, len(train_x[0])])
    y = tf.placeholder('float')

    train_neural_network(x, train_x, train_y, test_x, test_y, y)
    print("Total Elapsed Time = %.8f seconds" % (time.time() - begin))