import pickle, sys, os
import logging
from gensim.models import Word2Vec
from preprocess import util


def train_w2v_model():
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
    model.save("word2vecmodel")

    return model

if __name__ == '__main__':
    isfilesaved = False
    fn = input("Enter the pickle name: ")
    doesfileexist = True
    while not os.path.isfile(fn):
        fn = input("File does not exist, would you like to reprocess data (y/n)?")
        if fn == 'y':
            doesfileexist = False
            sentences = util.prepreprocessdata()
        else:
            fn = input("Enter filename?")

    if doesfileexist:
        sentences = pickle.load(open(fn, "rb"))
    f = open(fn)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

    # stpl: something to play with
    # todo stpl max_features
    # vectorizer = CountVectorizer(analyzer='word',
    #                              tokenizer=None,
    #                              preprocessor=None,
    #                              stop_words=None)

    model = train_w2v_model()

    # model.init_sims(replace=True)
