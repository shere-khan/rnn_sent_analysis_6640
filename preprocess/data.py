import pickle, sys, os
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

def openpickle():
    doesfileexist = True
    if input("Load data pickle (y/n)?") == 'y':
        fn = input("Enter the pickle name: ")
        while not os.path.isfile(fn):
            fn = input("File does not exist, would you like to reprocess data (y/n)?")
            if fn == 'y':
                doesfileexist = False
                sentences = util.prepreprocessdata()
            else:
                fn = input("Enter filename?")
    else: # process file
        sentences = None
        fn = input("Would you like to process data (y/n)?")
        if fn == 'y':
            doesfileexist = False
            sentences = util.prepreprocessdata()
        else:
            exit(0)


    if doesfileexist:
        sentences = pickle.load(open(fn, "rb"))

    return sentences

def openw2v():
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


if __name__ == '__main__':
    isfilesaved = False
    sentences = openpickle()
    dotrainw2v = input("Would you like to train w2v model (y/n)?")
    if dotrainw2v == 'y':
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                            level=logging.INFO)
        model = openw2v()
        print(model.doesnt_match("man woman child kitchen".split()))
        print(model.most_similar(positive=['woman', 'king'], negative=['man'], topn=10))
        # print(type(model.syn0))
        # print(model.syn0.shape)
    else:
        exit(0)

    # stpl: something to play with
    # todo stpl max_features
    # vectorizer = CountVectorizer(analyzer='word',
    #                              tokenizer=None,
    #                              preprocessor=None,
    #                              stop_words=None)

    # model.init_sims(replace=True)

