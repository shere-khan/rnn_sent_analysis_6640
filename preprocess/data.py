import pickle, sys, os, random
import logging
from gensim.models import Word2Vec
from preprocess import util
from keras import backend


def train_w2v_model(sentences):
    num_features = 300  # Word vector dimensionality
    min_word_count = 40  # Minimum word count
    num_workers = 8  # Number of threads to run in parallel
    context = 10  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words

    print("W2V training...")
    model = Word2Vec(sentences, workers=num_workers, size=num_features,
                     min_count=min_word_count, window=context, sample=downsampling,
                     seed=1)

    # Save the model
    model.save(input("Enter model name: "))

    return model

def openw2v(sentences):
    doesfileexist = True
    model = None
    fn = input("Enter w2v model name: ")
    while not os.path.isfile(fn):
        fn = input("File does not exist, would you like to reprocess data (y/n)?")
        if fn == 'y':
            doesfileexist = False
            model = train_w2v_model(sentences)
        else:
            fn = input("Enter filename?")

    if doesfileexist:
        model = Word2Vec.load("word2vecmodel")

    return model

def prepare_w2v_train():
    neg_train_reviews = pickle.load(
        open(input("Enter name of neg training data pickle: "), "rb"))
    pos_train_reviews = pickle.load(
        open(input("Enter name of pos training data pickle: "), "rb"))
    neg_test_reviews = pickle.load(
        open(input("Enter name of neg test data pickle: "), "rb"))
    pos_test_reviews = pickle.load(
        open(input("Enter name of pos test data pickle: "), "rb"))
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

    neg_training_sents = [s[:-2] for s in neg_train_reviews]
    pos_training_sents = [s[:-2] for s in pos_train_reviews]
    neg_test_sents = [s[:-2] for s in neg_test_reviews]
    pos_test_sents = [s[:-2] for s in pos_test_reviews]

    sents = list()
    neg_training_sents.extend(pos_training_sents)
    sents.extend(neg_test_sents)
    sents.extend(pos_test_sents)
    random.shuffle(sents)

    return sents

if __name__ == '__main__':
    isfilesaved = False
    option = input("What would you like to do?")
    print("1: Process data\n2: Train w2vec model\n3: Train RNN")
    if option == '1':
        util.prepreprocessdata()
    elif option == '2':
        sents = prepare_w2v_train()
        train_w2v_model(sents)

        # model = openw2v()
        # print(model.doesnt_match("man woman child kitchen".split()))
        # print(model.most_similar(positive=['woman', 'king'], negative=['man'], topn=10))
    elif option == '3':
        neg_training_data = input("Enter name of neg training data pickle: ")
        pos_training_data = input("Enter name of pos training data pickle: ")
        trainingdata = util.combinedata(neg_training_data, pos_training_data)
        training_sents = [review[:-2] for review in trainingdata]
        training_labels = [review[-1] for review in trainingdata]

        neg_test_data = input("Enter name of neg training data pickle: ")
        pos_test_data = input("Enter name of pos training data pickle: ")
        testdata = util.combinedata(neg_test_data, pos_test_data)
        test_sents = [review[:-2] for review in testdata]
        test_labels = [review[-1] for review in testdata]

    else:
        exit(0)

    # stpl: something to play with
    # todo stpl max_features
    # vectorizer = CountVectorizer(analyzer='word',
    #                              tokenizer=None,
    #                              preprocessor=None,
    #                              stop_words=None)

    # model.init_sims(replace=True)
