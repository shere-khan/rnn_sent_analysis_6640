import pickle, os, numpy as np
import logging, pylab as plt
import plotly as py
from gensim.models import Word2Vec
import util
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train_w2v_model(sentences):
    num_features = 300  # Word vector dimensionality
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

def ranForestClassifier(training, labels, n_est=100):
    rf = RandomForestClassifier(n_est)
    rf.fit(training, labels)

    if input("Would you like to save classifier (y/n)?") == 'y':
        with open(input('filename? '), 'wb') as f:
            pickle.dump(rf, f)

    return rf

def ranforest():
    model = pickle.load(open("data/w2v", "rb"))

    # Turn training reviews into vector averages
    train_reviews = pickle.load(open("data/train.p", "rb"))
    train_rev_avgs, train_labels = util.create_review_avgs_and_labels(train_reviews,
                                                                      model)
    # Turn test reviews into vector averages
    test_reviews = pickle.load(open("data/test.p", "rb"))
    test_rev_avgs, test_labels = util.create_review_avgs_and_labels(test_reviews,
                                                                    model)
    # Run RF classifier
    rf = ranForestClassifier(train_rev_avgs, train_labels)

    # Test RF Classifer and get accuracy
    predictions = rf.predict(test_rev_avgs)
    testacc = accuracy_score(test_labels, predictions)

    print("Test acc: {0}".format(testacc))

def recursive_nn():
    # model = pickle.load(open(input("Enter model name: "), "rb"))
    model = pickle.load(open("data/w2v", "rb"))

    # Turn training reviews into word vectors
    train_reviews = pickle.load(open("data/train.p", "rb"))
    training_word_vecs, trainlabels = util.transform_reviews_to_word_vectors(
        train_reviews, model)

    # Turn test reviews into word vectors
    test_reviews = pickle.load(open("data/test.p", "rb"))
    test_word_vecs, testlabels = util.transform_reviews_to_word_vectors(
        test_reviews, model)

    # Truncate sequences
    seq_length = 550
    training_word_vecs = [s[:seq_length] for s in test_word_vecs]
    test_word_vecs = [s[:seq_length] for s in test_word_vecs]

    # Pad sequences
    util.pad_sequences(training_word_vecs, seq_length, model.vector_size)
    util.pad_sequences(test_word_vecs, seq_length, model.vector_size)
    exit(0)

    # rnn.train(trainsents, testsents, timesteps=seq_length)

def plothist():
    dist = [len(s) for s in []]
    # dist = [len(s) for s in testsents]
    l = np.random.randn(10)
    print('dkf')

    plt.hist(np.array(dist))
    plt.xlabel("Lengths")
    plt.ylabel("Freq")

    py.tools.set_credentials_file(username='jbarry', api_key='NxzNmvLGLfOXz9xjF2HI')

    fig = plt.gcf()
    ploturl = py.plotly.plot_mpl(fig, filename='nlp.6640.rnn.seqeunce_length.dist')

def trainw2voption():
    train_reviews = pickle.load(
        open(input("Enter name of training data pickle: "), "rb"))
    test_reviews = pickle.load(
        open(input("Enter name of test data pickle: "), "rb"))
    sents = [r[0] for r in train_reviews]
    sents.extend([r[0] for r in test_reviews])
    train_w2v_model(sents)

if __name__ == '__main__':
    isfilesaved = False
    print("What would you like to do?")
    print("1: Process data\n2: Train w2vec model\n3: Train RNN"
          "\n4: Random Forest\n5: Plot hist of sequence lengths\n")
    option = input("input: ")
    if option == '1':
        util.prepreprocessdata(int(input("Cap: ")))
    elif option == '2':
        trainw2voption()
    elif option == '3':
        recursive_nn()
    elif option == '4':  # Use RF classifier
        ranforest()
    elif option == '5':  # plot histogram of sequence lengths
        plothist()
    elif option == '6':
        train = pickle.load(open("data/train.p", "rb"))
        test = pickle.load(open("data/test.p", "rb"))


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
