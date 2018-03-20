import pickle, os, rnn
import logging
from gensim.models import Word2Vec
from preprocess import util


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
    train_reviews = pickle.load(
        open(input("Enter name of training data pickle: "), "rb"))
    test_reviews = pickle.load(
        open(input("Enter name of test data pickle: "), "rb"))

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

    sents = util.extract_sentences_and_flatten(train_reviews)
    sents.extend(util.extract_sentences_and_flatten(test_reviews))

    return sents

if __name__ == '__main__':
    isfilesaved = False
    print("What would you like to do?")
    print("1: Process data\n2: Train w2vec model\n3: Train RNN")
    option = input("input: ")
    if option == '1':
        util.prepreprocessdata(input("Cap: "))
    elif option == '2':
        sents = prepare_w2v_train()
        train_w2v_model(sents)
    elif option == '3':
        train_reviews = pickle.load(
            open(input("Enter name of training data pickle: "), "rb"))

        # training_sents, training_label = util.split_sents_and_labels(train_reviews)

        # test_reviews = pickle.load(
        #     open(input("Enter name of test data pickle: "), "rb"))
        # test_sents, test_label = util.split_sents_and_labels(train_reviews)

        # rnn.train()
    else:
        exit(0)

    # stpl: something to play with
    # todo stpl max_features
    # vectorizer = CountVectorizer(analyzer='word',
    #                              tokenizer=None,
    #                              preprocessor=None,
    #                              stop_words=None)

    # model.init_sims(replace=True)

    # model = openw2v()
    # print(model.doesnt_match("man woman child kitchen".split()))
    # print(model.most_similar(positive=['woman', 'king'], negative=['man'], topn=10))
