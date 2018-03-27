import numpy as np
import tensorflow as tf
import os, random, pickle, time, util
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer


n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 2
batch_size = 100

lemmatizer = WordNetLemmatizer()
hm_lines = 100000000

def tok_and_lem(data, lexicon):
    for d in data:
        # words = word_tokenize(l.lower())
        words = [w for w in d[0] if w not in util.stops]
        words = [lemmatizer.lemmatize(i) for i in words]
        lexicon += list(set(words))

def create_lexicon(train, test):
    files, lexicon = [], []
    start = time.time()

    tok_and_lem(train, lexicon)
    tok_and_lem(test, lexicon)

    w_counts = Counter(lexicon)

    l2 = []
    # todo create his with freqs to decide
    for w in w_counts:
        if 250 > w_counts[w] > 1:
            l2.append(w)


    print("Lexicon size = ", len(lexicon))
    print("Creating lexicon took %.8f seconds" % (time.time() - start))

    inp = input("Enter pickle name: ")
    print("entered")
    pickle.dump(lexicon, open(inp, "wb"))

    return lexicon

def createfeats(data, lexicon):
    featureset = []
    # todo change here to do averages
    for d in data:
        words = d[0]
        # words = [w for w in words if w not in util.stops]
        current_words = [lemmatizer.lemmatize(i) for i in words]
        features = np.zeros(len(lexicon))

        for word in current_words:
            if word in lexicon:
                index_value = lexicon.index(word)
                features[index_value] += 1

        features = list(features)
        featureset.append([features, d[1]])

    return featureset

def create_feature_sets_and_labels(pos, neg, testpos, testneg, test_size=0.1):
    start = time.time()

    training = pickle.load(open(input('training data name: '), "rb"))
    test = pickle.load(open(input('test data name: '), "rb"))

    if input("load lexicon pickle (y/n)?") == "y":
        lexicon = pickle.load(open(input('lexicon name: '), "rb"))
    else:
        lexicon = create_lexicon(training, test)

    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None,
                                 stop_words=None, max_features=5000)


    trainclean, trainy = util.extract_clean_reviews(training)
    testclean, testy = util.extract_clean_reviews(test)

    print("vectorizing training data")
    trainx = vectorizer.fit_transform(trainclean).toarray()
    print("vectorizing test data")
    testx = vectorizer.fit_transform(testclean).toarray()

    # print("Creating training features")
    # trainingfeats = createfeats(training, lexicon)
    # print("Creating test features")
    # testfeats = createfeats(test, lexicon)
    #
    # trainingfeats = np.array(trainingfeats)
    # train_x = list(trainingfeats[:, 0])
    # train_y = list(trainingfeats[:, 1])
    #
    # testfeats = np.array(testfeats)
    # test_x = list(testfeats [:, 0])
    # test_y = list(testfeats [:, 1])

    print("Creating Sets and Labels took %.8f seconds" % (time.time() - start))
    return trainx, trainy, testx,testy

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

    hm_epochs = 14
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

    pickle.dump(train_x, open('train_x.p', 'wb'))
    pickle.dump(train_y, open('train_y.p', 'wb'))
    pickle.dump(test_x, open('test_x.p', 'wb'))
    pickle.dump(test_y, open('test_y.p', 'wb'))

    x = tf.placeholder('float', [None, len(train_x[0])])
    y = tf.placeholder('float')

    train_neural_network(x, train_x, train_y, test_x, test_y, y)
    print("Total Elapsed Time = %.8f seconds" % (time.time() - begin))
