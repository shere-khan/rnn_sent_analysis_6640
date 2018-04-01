import pickle, time
import logging
from gensim.models import Word2Vec
import util
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train_w2v_model(sentences):
    num_features = 1000  # Word vector dimensionality
    min_word_count = 40  # Minimum word count
    num_workers = 8  # Number of threads to run in parallel
    context = 10  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    print("W2V training...")
    model = Word2Vec(sentences, workers=num_workers, size=num_features,
                     min_count=min_word_count, window=context, sample=downsampling,
                     seed=1)

    # Save the model
    model.save("data/rf/w2vrf")

    return model

def ranForestClassifier(training, labels, n_est=100):
    rf = RandomForestClassifier(n_est)
    rf.fit(training, labels)

    return rf

def ranforest():
    start = time.time()
    model = Word2Vec.load("data/rf/w2vrf")

    # Turn training reviews into vector averages
    train_reviews = pickle.load(open("data/rf/train.out", "rb"))
    train_rev_avgs, train_labels = util.create_review_avgs_and_labels(train_reviews,
                                                                      model)
    # Turn test reviews into vector averages
    test_reviews = pickle.load(open("data/rf/test.out", "rb"))
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
    train_reviews = pickle.load(
        open("data/rf/train.out", "rb"))
    test_reviews = pickle.load(
        open("data/rf/test.out", "rb"))
    sents = [r[0] for r in train_reviews]
    sents.extend([r[0] for r in test_reviews])
    train_w2v_model(sents)

if __name__ == '__main__':
    isfilesaved = False
    print("What would you like to do?")
    print("1: Process data\n2: Train w2vec model\n3: Random Forest Classifier\n\n")
    option = input("input: ")
    if option == '1':
        captrain = int(input("Cap train: "))
        captest = int(input("Cap test: "))
        util.prepreprocessdata(captrain, captest)
    elif option == '2':
        trainw2voption()
    elif option == '3':  # Use RF classifier
        ranforest()
