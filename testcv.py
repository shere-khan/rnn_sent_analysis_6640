import util, pickle, numpy as np, time, math, random, logging
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
from gensim.models import Word2Vec, KeyedVectors
from nltk import word_tokenize
from bs4 import BeautifulSoup
from nltk.stem.wordnet import WordNetLemmatizer
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
hm_lines = 100000000
num_feats = 3000

total_batches = int(50000 / batch_size)

tf_log = 'tf.log'

def create_lexicon():
    data = util.extract_raw_data(['/home/justin/pycharmprojects/rnn_sent_analysis_6640'
                                  '/data/processed/'], cap=None)
    train = [x[0] for x in data]
    vectorizer = CountVectorizer(analyzer="word", max_features=num_feats)
    vectorizer.fit(train)
    pickle.dump(vectorizer, open("data/w2v", "wb"))

def reformatdata(cap=None, stops=False):
    data = util.extract_raw_data(filenames, cap=cap)
    random.shuffle(data)

    p = math.ceil(len(data) * .95)
    training = data[:p]
    test = data[p:]
    lemmatizer = WordNetLemmatizer()

    with open("data/processed/training.out", "w") as f:
        for d in training:
            review = BeautifulSoup(d[0], "html5lib").get_text()
            review = util.remove_emoji_and_nums(review)
            words = review.split()
            words = words[:-1]
            if stops:
                words = [w for w in words if w not in util.stops]
            words = [lemmatizer.lemmatize(w.lower()) for w in words]
            sent = " ".join(words)
            label = 1 if d[1][0] == 1 else 0
            f.write("{0}::::{1}\n".format(sent, label))

    print("reformat test")
    with open("data/processed/test.out", "w") as f:
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
                      'weight': tf.Variable(tf.random_normal([num_feats, n_nodes_hl1])),
                      'bias': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'f_fum': n_nodes_hl2,
                      'weight': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'bias': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    output_layer = {'f_fum': None,
                    'weight': tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                    'bias': tf.Variable(tf.random_normal([n_classes])), }

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)

    output = tf.matmul(l2, output_layer['weight']) + output_layer['bias']

    return output

def train_neural_network(x, y, model):
    start2 = time.time()
    prediction = neural_network_model(x)
    saver = tf.train.Saver()
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 20
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
            with open('data/processed/training.out', buffering=20000,
                      encoding='latin-1') as f:
                batch_x = []
                batch_y = []
                batches_run = 0
                for line in f:
                    review = line.split("::::")
                    label = [1, 0] if review[1][:-1] == '1' else [0, 1]
                    sentence = review[0]
                    features = model.transform([sentence])

                    line_x = features.toarray().tolist()[0]
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

        test_x, test_y = get_test_set_data("data/processed/test.out", model)

        print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))

    print("Training NN took %.8f seconds" % (time.time() - start2))

def get_test_set_data(fn, model):
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

def train_network_w2v():
    x = tf.placeholder('float')
    y = tf.placeholder('float')

    model = Word2Vec.load("data/w2v_nn")
    train_neural_network_w2v(x, y, model)

    # print("Total Elapsed Time = %.2f seconds" % (time.time() - begin))

def train_network():
    x = tf.placeholder('float')
    y = tf.placeholder('float')

    model = pickle.load(open("data/w2v", "rb"))
    train_neural_network(x, y, model)

    # print("Total Elapsed Time = %.2f seconds" % (time.time() - begin))

def create_review_avgs_and_labels(reviews, model):
    vocab = set(model.wv.vocab)
    features = list()
    for rev in reviews:
        fvec = np.zeros(model.vector_size, dtype=np.float32)
        wordct = 0
        words = rev.split()
        for w in words:
            if w in vocab:
                wordct += 1
                fvec = np.add(fvec, model[w])
        fvec = np.divide(fvec, wordct)
        features.append(fvec)

    return np.array(features)

def train_w2v_model(sentences):
    min_word_count = 5  # Minimum word count
    num_workers = 8  # Number of threads to run in parallel
    context = 10  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    print("W2V training...")
    model = Word2Vec(sentences, workers=num_workers, size=num_feats,
                     min_count=min_word_count, window=context, sample=downsampling,
                     seed=1)

    # Save the model
    # model.save(input("Enter model name: "))
    model.save("data/w2v_nn")

    return model

def train_neural_network_w2v(x, y, model):
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
            with open('data/processed/training.out', buffering=20000,
                      encoding='latin-1') as f:
                batch_x = []
                batch_y = []
                batches_run = 0
                for line in f:
                    review = line.split("::::")
                    label = [1, 0] if review[1][:-1] == '1' else [0, 1]
                    sentence = review[0]
                    batch_x.append(sentence)
                    batch_y.append(label)
                    if len(batch_x) >= batch_size:
                        train_x = create_review_avgs_and_labels(batch_x, model)
                        _, c = sess.run([optimizer, cost],
                                        feed_dict={x: train_x,
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

        test_x, test_y = get_test_set_data_w2v("data/processed/test.out", model)

        print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))

    print("Training NN took %.8f seconds" % (time.time() - start2))

def get_test_set_data_w2v(fn, model):
    reviews = []
    labels = []
    with open(fn, "r") as f:
        for line in f:
            review = line.split("::::")
            sent = review[0]
            label = [1, 0] if review[1][:-1] == '1' else [0, 1]
            reviews.append(sent)
            labels.append(label)

    features = create_review_avgs_and_labels(reviews, model)
    return features, np.array(labels)


if __name__ == '__main__':
    # BOW
    # reformatdata(stops=True)
    # create_lexicon()
    # train_network()

    # W2V
    # reformatdata(cap=None, stops=False)
    # data = util.extract_raw_data(['/home/justin/pycharmprojects/rnn_sent_analysis_6640'
    #                               '/data/processed/'], cap=None)
    # sents = [x[0].split("::::")[0].split() for x in data]
    # model = train_w2v_model(sents)
    train_network_w2v()
