import util, pickle, numpy as np, time
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
from nltk import word_tokenize
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


filenames = ['/home/justin/pycharmprojects/rnn_sent_analysis_6640/reviews/train/neg/',
             '/home/justin/pycharmprojects/rnn_sent_analysis_6640/reviews/train/pos/',
             '/home/justin/pycharmprojects/rnn_sent_analysis_6640/reviews/test/neg/',
             '/home/justin/pycharmprojects/rnn_sent_analysis_6640/reviews/test/pos/']

n_classes = 2
batch_size = 100
n_nodes_hl1 = 750
n_nodes_hl2 = 750
n_nodes_hl3 = 750
lemmatizer = WordNetLemmatizer()
hm_lines = 100000000

total_batches = int(1600000 / batch_size)

tf_log = 'tf.log'

def vectorizeexample():
    data = util.extract_raw_data(['/home/justin/pycharmprojects/rnn_sent_analysis_6640'
                                  '/data/processed/'])

    vectorizer = CountVectorizer(analyzer="word", preprocessor=None, max_features=1000)

    train = [x[0] for x in data]

    vectorizer.fit_transform(train)
    print(vectorizer.vocabulary_)
    pickle.dump(vectorizer, open("data/bow_mod1000", "wb"))

def reformatdata(stops=False):
    data = util.extract_raw_data(filenames, cap=None)

    lemmatizer = WordNetLemmatizer()
    with open("data/data1.out", "w") as f:
        for d in data:
            review = BeautifulSoup(d[0], "html5lib").get_text()
            review = util.remove_emoji_and_nums(review)
            words = word_tokenize(review)
            if stops:
                words = [lemmatizer.lemmatize(w) for w in words if w not in util.stops]
            sent = " ".join(words)
            label = 1 if d[1][0] == 1 else 0
            f.write("{0} {1}\n".format(sent, label))

def neural_network_model(data, train_x):
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
    return output

def train_neural_network(x, y, train_x, train_y, test_x, test_y, saver):
    start2 = time.time()
    prediction = neural_network_model(x, train_x)
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 14
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        try:
            epoch = int(open(tf_log, 'r').read().split('\n')[-2]) + 1
            print('STARTING:', epoch)
        except:
            epoch = 1

        while epoch <= hm_epochs:
            if epoch != 1:
                saver.restore(sess, "model.ckpt")
            epoch_loss = 1
            with open('data/bow_mod1000', 'rb') as f:
                lexicon = pickle.load(f)
            with open('data/processed/data.out', buffering=20000, encoding='latin-1') as f:
                batch_x = []
                batch_y = []
                batches_run = 0
                for line in f:
                    label = line.split()[0]
                    tweet = line.split(':::')[1]
                    current_words = word_tokenize(tweet.lower())
                    current_words = [lemmatizer.lemmatize(i) for i in current_words]

                    features = np.zeros(len(lexicon))

                    for word in current_words:
                        if word.lower() in lexicon:
                            index_value = lexicon.index(word.lower())
                            # OR DO +=1, test both
                            features[index_value] += 1

                    line_x = list(features)
                    line_y = eval(label)
                    batch_x.append(line_x)
                    batch_y.append(line_y)
                    if len(batch_x) >= batch_size:
                        _, c = sess.run([optimizer, cost],
                                        feed_dict={x: np.array(batch_x),
                                                   y: np.array(batch_y)})
                        epoch_loss += c
                        batch_x = []
                        batch_y = []
                        batches_run += 1
                        print('Batch run:', batches_run, '/', total_batches, '| Epoch:',
                              epoch, '| Batch Loss:', c, )

            saver.save(sess, "model.ckpt")
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
            with open(tf_log, 'a') as f:
                f.write(str(epoch) + '\n')
            epoch += 1

            # START OTHER CODE
    #         epoch_loss = 0
    #
    #         i = 0
    #         while i < len(train_x):
    #             start = i
    #             end = i + batch_size
    #
    #             batch_x = np.array(train_x[start:end])
    #             batch_y = np.array(train_y[start:end])
    #
    #             _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
    #             epoch_loss += c
    #             i += batch_size
    #
    #         print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)
    #
    #     correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    #     accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    #
    #     print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))
    print("Training NN took %.8f seconds" % (time.time() - start2))

# def train_network():
#     train_x, train_y, test_x, test_y = (pos, neg, testpos, testneg)
#
#     x = tf.placeholder('float', [None, len(train_x[0])])
#     y = tf.placeholder('float')
#
#     train_neural_network(x, train_x, train_y, test_x, test_y, y)
#
#     print("Total Elapsed Time = %.2f seconds" % (time.time() - begin))

if __name__ == '__main__':
    # vectorizeexample()
    reformatdata(True)
    # saver = tf.train.Saver()

    # train_network()
