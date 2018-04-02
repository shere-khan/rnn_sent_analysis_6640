import pickle, time
import logging
from gensim.models import Word2Vec
import util
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train_w2v_model(sentences):
    num_feats = 1000
    min_ct= 40
    num_workers = 8
    cxt = 10
    dwnsamp = 1e-3

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    print("W2V training...")
    model = Word2Vec(sentences, workers=num_workers, size=num_feats,
                     min_count=min_ct, window=cxt, sample=dwnsamp,
                     seed=1)

    # Save the model
    model.save("w2vrf")

    return model

def ranForestClassifier(training, labels, n_est=100):
    rf = RandomForestClassifier(n_est)
    rf.fit(training, labels)

    return rf

def ranforest():
    start = time.time()
    model = Word2Vec.load("w2vrf")

    # Turn training reviews into vector averages
    train_reviews = pickle.load(open("rftrain.out", "rb"))
    train_rev_avgs, train_labels = util.create_review_avgs_and_labels(train_reviews,
                                                                      model)
    # Turn test reviews into vector averages
    test_reviews = pickle.load(open("rftest.out", "rb"))
    test_rev_avgs, test_labels = util.create_review_avgs_and_labels(test_reviews,
                                                                    model)
    # Run RF classifier
    rf = ranForestClassifier(train_rev_avgs, train_labels)
    print("Time: {0}".format(time.time() - start))

    # Test RF Classifer and get accuracy
    predictions = rf.predict(test_rev_avgs)
    testacc = accuracy_score(test_labels, predictions)

    print("Test acc: {0}".format(testacc))

def trainw2voption():
    train_reviews = pickle.load(open("rftrain.out", "rb"))
    test_reviews = pickle.load(open("rftest.out", "rb"))
    sents = [r[0] for r in train_reviews]
    sents.extend([r[0] for r in test_reviews])
    train_w2v_model(sents)

if __name__ == '__main__':
    util.prepreprocessdata(-1, -1)
    trainw2voption()
    ranforest()
