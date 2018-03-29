import util, pickle, numpy as np, time, math, random
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
batch_size = 5000
n_nodes_hl1 = 1000
n_nodes_hl2 = 1000
n_nodes_hl3 = 1000
lemmatizer = WordNetLemmatizer()
hm_lines = 100000000

total_batches = int(50000 / batch_size)

tf_log = 'tf.log'

def create_lexicon():
    data = util.extract_raw_data(['/home/justin/pycharmprojects/rnn_sent_analysis_6640'
                                  '/data/processed/'], cap=None)
    class LemmaTokenizer(object):
        def __init__(self):
            self.wnl = WordNetLemmatizer()
        def __call__(self, doc):
            return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

    vectorizer = CountVectorizer(analyzer="word", preprocessor=lemmatizer,
                                 stop_words=util.stops, max_features=2000)

    train = [x[0] for x in data]

    vectorizer.fit_transform(train)
    for d in data:
        feats = vectorizer.transform([d[0]])
        for f in feats.toarray():
            print(f)

def reformatdata(stops=False):
    data = util.extract_raw_data(filenames, cap=None)
    random.shuffle(data)

    p = math.ceil(len(data) * .95)
    training = data[:p]
    test = data[p:]
    lemmatizer = WordNetLemmatizer()

    with open("data/training.out", "w") as f:
        for d in training:
            review = BeautifulSoup(d[0], "html5lib").get_text()
            review = util.remove_emoji_and_nums(review)
            words = word_tokenize(review)
            if stops:
                words = [lemmatizer.lemmatize(w) for w in words if w not in util.stops]
            sent = " ".join(words)
            label = 1 if d[1][0] == 1 else 0
            f.write("{0}::::{1}\n".format(sent, label))

    print("reformat test")
    with open("data/test.out", "w") as f:
        for d in test:
            review = BeautifulSoup(d[0], "html5lib").get_text()
            review = util.remove_emoji_and_nums(review)
            words = word_tokenize(review)
            if stops:
                words = [lemmatizer.lemmatize(w) for w in words if w not in util.stops]
            sent = " ".join(words)
            label = 1 if d[1][0] == 1 else 0
            f.write("{0}::::{1}\n".format(sent, label))

def neural_network_model(data):
    hidden_1_layer = {'f_fum': n_nodes_hl1,
                      'weight': tf.Variable(tf.random_normal([1000, n_nodes_hl1])),
                      'bias': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'f_fum': n_nodes_hl2,
                      'weight': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'bias': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    output_layer = {'f_fum': None,
                    'weight': tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                    'bias': tf.Variable(tf.random_normal([n_classes])), }

    # hidden_1_layer = {
    #     'weights': tf.Variable(tf.random_normal([1000, n_nodes_hl1])),
    #     'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    #
    # hidden_2_layer = {
    #     'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
    #     'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    #
    # hidden_3_layer = {
    #     'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
    #     'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    # output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
    #                 'biases': tf.Variable(tf.random_normal([n_classes])), }

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)

    # l3 = tf.add(tf.matmul(l2, hidden_3_layer['weight']), hidden_3_layer['bias'])
    # l3 = tf.nn.relu(l3)

    output = tf.matmul(l2, output_layer['weight']) + output_layer['bias']

    return output

def train_neural_network(x, y, model):
    start2 = time.time()
    prediction = neural_network_model(x)
    saver = tf.train.Saver()
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 15
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        try:
            epoch = int(open(tf_log, 'r').read().split('\n')[-2]) + 1
            print('STARTING:', epoch)
        except:
            epoch = 1

        while epoch <= hm_epochs:
            if epoch != 1:
                saver.restore(sess, "data/model.ckpt")
            epoch_loss = 1
            with open('data/bow_mod1000', 'rb') as f:
                lexicon = pickle.load(f)
            with open('data/training.out', buffering=20000, encoding='latin-1') as f:
                batch_x = []
                batch_y = []
                batches_run = 0
                for line in f:
                    review = line.split("::::")
                    label = [1, 0] if review[1][:-1] == '1' else [0, 1]
                    sentence = review[0]
                    features = model.transform([sentence])

                    line_x = features.toarray().tolist()[0]
                    # line_x = list(features)
                    # line_y = eval(label)
                    batch_x.append(line_x)
                    batch_y.append(label)
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

            saver.save(sess, "data/model.ckpt")
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
            with open(tf_log, 'a') as f:
                f.write(str(epoch) + '\n')
            epoch += 1
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        test_x, test_y = get_test_set("data/test.out", model)

        print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))

    print("Training NN took %.8f seconds" % (time.time() - start2))

def get_test_set(fn, model):
    reviews = []
    labels = []
    with open(fn, "r") as f:
        for line in f:
            review = line.split("::::")
            sent = review[0]
            label = [1, 0] if review[1][:-1] == '1' else [0, 1]
            features = model.transform([sent])
            reviews.append(features.toarray().tolist()[0 ])
            labels.append(label)

    return np.array(reviews), np.array(labels)


def train_network():
    # x = tf.placeholder('float', [None, len(train_x[0])])
    x = tf.placeholder('float')
    y = tf.placeholder('float')

    model = pickle.load(open("data/bow_mod1000", "rb"))
    train_neural_network(x, y, model)

    # print("Total Elapsed Time = %.2f seconds" % (time.time() - begin))

if __name__ == '__main__':
    create_lexicon()
    # reformatdata(True)

    # train_network()
