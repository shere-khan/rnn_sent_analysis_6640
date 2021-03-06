import util, pickle, numpy as np, time, math, random, logging, os, errno
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score
from bs4 import BeautifulSoup
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize



num_classes = 2
batch_size = 5000
num_nodes_hl1 = 1000
num_nodes_hl2 = 1000
num_feats = 2000

total_num_batches = int(50000 / batch_size)

logfile = 'tf.log'

def reformatdata(filenames, cap=None, stops=False):
    data = util.extract_raw_data(filenames, cap=cap)
    random.shuffle(data)

    p = math.ceil(len(data) * .95)
    training = data[:p]
    test = data[p:]
    lemmatizer = WordNetLemmatizer()

    if not os.path.exists(os.path.dirname("data/processed/training.out")):
        try:
            os.makedirs(os.path.dirname("data/processed/training.out"))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open("data/processed/training.out", "w+") as f:
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
            f.write("{0}+:::{1}\n".format(sent, label))

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
            f.write("{0}+:::{1}\n".format(sent, label))

def neural_network_model(data):
    hidden_1_layer = {'f_fum': num_nodes_hl1,
                      'weight': tf.Variable(tf.random_normal([num_feats, num_nodes_hl1])),
                      'bias': tf.Variable(tf.random_normal([num_nodes_hl1]))}

    hidden_2_layer = {'f_fum': num_nodes_hl2,
                      'weight': tf.Variable(tf.random_normal([num_nodes_hl1, num_nodes_hl2])),
                      'bias': tf.Variable(tf.random_normal([num_nodes_hl2]))}

    output_layer = {'f_fum': None,
                    'weight': tf.Variable(tf.random_normal([num_nodes_hl2, num_classes])),
                    'bias': tf.Variable(tf.random_normal([num_classes])), }

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)

    output = tf.matmul(l2, output_layer['weight']) + output_layer['bias']

    return output

def get_test_set_data(fn, model):
    reviews = []
    labels = []
    with open(fn, "r") as f:
        for line in f:
            review = line.split("+:::")
            sent = review[0]
            label = [1, 0] if review[1][:-1] == '1' else [0, 1]
            features = model.transform([sent])
            reviews.append(features.toarray().tolist()[0])
            labels.append(label)

    return np.array(reviews), np.array(labels)

def train_network_w2v():
    x = tf.placeholder('float')
    y = tf.placeholder('float')

    model = Word2Vec.load("data/w2v_nn")
    train_neural_network_w2v(x, y, model)

def create_review_avgs_and_labels(reviews, model):
    vocab = set(model.wv.vocab)
    features = list()
    for i, rev in enumerate(reviews):
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
    model.save("data/w2v_nn")

    return model

def train_neural_network_w2v(x, y, model):
    start2 = time.time()
    est = neural_network_model(x)
    sav = tf.train.Saver()
    cst = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=est, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cst)

    num_epochs = 20
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        try:
            epc = int(open(logfile, 'r').read().split('\n')[-2]) + 1
            print('STARTING:', epc)
        except:
            epc = 1

        while epc <= num_epochs:
            if epc != 1:
                sav.restore(sess, "data/model.ckpt")
            loss = 1
            with open('data/processed/training.out', buffering=20000) as f:
                set_x = []
                set_y = []
                batches_run = 0
                for line in f:
                    review = line.split("+:::")
                    label = [1, 0] if review[1][0] == '1' else [0, 1]
                    sentence = review[0]
                    set_x.append(sentence)
                    set_y.append(label)
                    if len(set_x) >= batch_size:
                        train_x = create_review_avgs_and_labels(set_x, model)
                        _, c = sess.run([optimizer, cst],
                                        feed_dict={x: train_x,
                                                   y: np.array(set_y)})
                        loss += c
                        set_x = []
                        set_y = []
                        batches_run += 1
                        print("Batch run:{0}/{1} | Epoch: {2} | Batch Loss: {3}"
                              .format(batches_run, total_num_batches, epc, c))

            sav.save(sess, "data/model.ckpt")
            print('Completed epoch {0} out of {1} loss = {2}'.format(epc, num_epochs,
                                                                     loss))
            with open(logfile, 'a') as f:
                f.write(str(epc) + '\n')
            epc += 1
        correct = tf.equal(tf.argmax(est, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        test_x, test_y = get_test_set_data_w2v("data/processed/test.out", model)

        print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))

    print("Training NN took %.8f seconds" % (time.time() - start2))

def get_test_set_data_w2v(fn, model):
    reviews = []
    labels = []
    with open(fn, "r") as f:
        for line in f:
            review = line.split("+:::")
            sent = review[0]
            label = [1, 0] if review[1][:-1] == '1' else [0, 1]
            reviews.append(sent)
            labels.append(label)

    features = create_review_avgs_and_labels(reviews, model)
    return features, np.array(labels)

def ranforest(sents, labels):
    model = Word2Vec.load("data/w2v_nn")

    # Turn sentences into vector avgs
    print("Create feats")
    features = create_review_avgs_and_labels(sents, model)

    # Divide feat and lab matrices into trianing set and test set
    p = math.ceil(len(features) * .95)
    train_x = features[:p]
    train_y = np.array(labels[:p])
    test_x = features[p:]
    test_y = np.array(labels[p:])

    # Run RF classifier
    rf = ranForestClassifier(train_x, train_y)

    # Test RF Classifer and get accuracy
    predictions = rf.predict(test_x)
    testacc = accuracy_score(test_y, predictions)

    print("Test acc: {0}".format(testacc))

def ranForestClassifier(training, labels, n_est=100):
    rf = RandomForestClassifier(n_est)
    rf.fit(training, labels)

    if input("Would you like to save classifier (y/n)?") == 'y':
        with open(input('filename? '), 'wb') as f:
            pickle.dump(rf, f)

    return rf

def getlabels(data):
    labels = []
    for d in data:
        val = d[0].split("+:::")[1][0]
        # print()
        labels.append(int(val))


    return labels

def extractdata(paths, cap=1, addlabels=False):
    l = []
    for path in paths:
        val = path.split("/")[-2]
        if addlabels:
            lab = [1, 0] if val == 'pos' else [0, 1]
        for i, file in enumerate(os.listdir(path)):
            if cap is None or i < cap:
                with open(path + file, "r") as f:
                    for line in f:
                        if addlabels:
                            l.append([line, lab])
                        else:
                            l.append([line])
            else:
                break

    return l

def getsents(data):
    sents = []
    for d in data:
        val = d[0].split("+:::")[0]
        sents.append(val)

    return sents

if __name__ == '__main__':
    # Prep data for W2VNN
    pos = input("Enter pos train review location: ")
    neg = input("Enter neg train review location: ")
    testpos = input("Enter pos test review location: ")
    testneg = input("Enter neg test review location: ")
    filenames = [neg, pos, testneg, testpos]
    reformatdata(filenames, cap=None, stops=False)
    data = util.extract_raw_data(['data/processed/'], cap=None)
    sents = [x[0].split("+:::")[0].split() for x in data]
    train_w2v_model(sents)
    train_network_w2v()
