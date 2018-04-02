import numpy as np
import tensorflow as tf
from collections import Counter
import os, nltk, random, pickle, time
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


classes = 2
batch = 100
nodes = 1000
lemmatizer = WordNetLemmatizer()

def lex(pos, neg):
    lexicon = []
    start = time.time()
    for f in os.listdir(pos):
        words = open(pos + f, 'r', encoding="utf8").readlines()
        words = words[0].split()
        for w in words:
            word = word_tokenize(w.lower())
            lexicon += word

    for f in os.listdir(neg):
        words = open(neg + f, 'r', encoding="utf8").readlines()
        words = words[0].split()
        for w in words:
            word = word_tokenize(w.lower())
            lexicon += word

    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    counts = Counter(lexicon)

    trimmed = []
    for w in counts:
        if (1000 > counts[w] > 75):
            trimmed.append(w)

    lexicon = {}
    for i in range(len(trimmed)):
        lexicon[trimmed[i]] = i

    print("Lexicon size = ", len(lexicon))
    print("Creating lexicon took %.2f seconds" % (time.time() - start))
    return lexicon

def vectorize(review, lexicon, clas):
    vector = []
    features = np.zeros(len(lexicon))
    f = open(review, 'r', encoding="utf8")
    words = f.readlines()
    words = words[0].split()
    for i in range(len(words)):
        word = word_tokenize(words[i].lower())
        words[i] = lemmatizer.lemmatize(word[0])

    for word in words:
        if (word in lexicon):
            features[lexicon[word]] += 1

    features = list(features)
    vector.append([features, clas])

    return vector

def prep_data(pos, neg, testpos, testneg):
    start = time.time()
    lexicon = lex(pos, neg)

    t2 = time.time()
    vectors = []
    for f in os.listdir(pos):
        vectors += vectorize(pos + f, lexicon, [1, 0])

    t1 = time.time()
    print("Adding positive training data took %.2f seconds" % (t1 - t2))

    for f in os.listdir(neg):
        vectors += vectorize(neg + f, lexicon, [0, 1])

    t2 = time.time()
    print("Adding negative training data took %.2f seconds" % (t2 - t1))

    random.shuffle(vectors)
    vectors = np.array(vectors)
    train_x = list(vectors[:, 0])
    train_y = list(vectors[:, 1])

    t2 = time.time()
    vectors = []
    for f in os.listdir(testpos):
        vectors += vectorize(testpos + f, lexicon, [1, 0])
    t1 = time.time()
    print("Adding positive test data took %.2f seconds" % (t1 - t2))

    for f in os.listdir(testneg):
        vectors += vectorize(testneg + f, lexicon, [0, 1])
    t2 = time.time()
    print("Adding negative test data took %.2f seconds" % (t2 - t1))

    random.shuffle(vectors)
    vectors = np.array(vectors)
    test_x = list(vectors[:, 0])
    test_y = list(vectors[:, 1])

    print("Creating Sets and Labels took %.2f seconds" % (time.time() - start))
    return train_x, train_y, test_x, test_y

def NN(data, train_x):
    layer1 = {'weights': tf.Variable(tf.random_normal([len(train_x[0]), nodes])),
              'biases': tf.Variable(tf.random_normal([nodes]))}

    output = {'weights': tf.Variable(tf.random_normal([nodes, classes])),
              'biases': tf.Variable(tf.random_normal([classes])), }

    layer1 = tf.add(tf.matmul(data, layer1['weights']), layer1['biases'])
    layer1 = tf.nn.relu(layer1)

    output = tf.matmul(layer1, output['weights']) + output['biases']
    return output

def Train_NN(x, train_x, train_y, test_x, test_y, y):
    start2 = time.time()
    model = NN(x, train_x)
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            loss = 0

            i = 0
            while i < len(train_x):
                start = i
                end = i + batch

                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                z, l = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                loss += l
                i += batch

            print('Epoch', epoch + 1, 'loss: %.4f' % loss)

        correct = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))
    print("Training NN took %.4f seconds" % (time.time() - start2))

if __name__ == '__main__':
    begin = time.time()
    pos = input("Enter pos train review location: ")
    neg = input("Enter neg train review location: ")
    testpos = input("Enter pos test review location: ")
    testneg = input("Enter neg test review location: ")

    train_x, train_y, test_x, test_y = prep_data(pos, neg, testpos, testneg)

    x = tf.placeholder('float', [None, len(train_x[0])])
    y = tf.placeholder('float')

    Train_NN(x, train_x, train_y, test_x, test_y, y)

    print("Total Elapsed Time = %.2f seconds" % (time.time() - begin))
