import util, pickle, numpy as np, time, math, random, logging, os
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score
from bs4 import BeautifulSoup
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def create_lexicon():
    data = util.extract_raw_data(['/home/justin/pycharmprojects/rnn_sent_analysis_6640'
                                  '/data/processed/'], cap=None)
    train = [x[0] for x in data]
    vectorizer = CountVectorizer(analyzer="word", max_features=num_feats)
    vectorizer.fit(train)
    pickle.dump(vectorizer, open("data/w2v", "wb"))

def train_neural_network(x, y, model):
    start2 = time.time()
    prediction = neural_network_model(x)
    saver = tf.train.Saver()
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 25
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        try:
            epoch = int(open(logfile, 'r').read().split('\n')[-2]) + 1
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
            with open(logfile, 'a') as f:
                f.write(str(epoch) + '\n')
            epoch += 1
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        test_x, test_y = get_test_set_data("data/processed/test.out", model)

        print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))

    print("Training NN took %.8f seconds" % (time.time() - start2))

def train_network():
    x = tf.placeholder('float')
    y = tf.placeholder('float')

    model = pickle.load(open("data/w2v", "rb"))
    train_neural_network(x, y, model)

    # print("Total Elapsed Time = %.2f seconds" % (time.time() - begin))

if __name__ == '__main__':
    # BOW
    # reformatdata(stops=True)
    # create_lexicon()
    train_network()

