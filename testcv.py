import util, pickle, numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

filenames = ['/home/justin/pycharmprojects/rnn_sent_analysis_6640/dataset/train/neg/',
             '/home/justin/pycharmprojects/rnn_sent_analysis_6640/dataset/train/pos/',
             '/home/justin/pycharmprojects/rnn_sent_analysis_6640/dataset/test/neg/',
             '/home/justin/pycharmprojects/rnn_sent_analysis_6640/dataset/test/pos/']

def vectorizeexample():
    data = util.extract_raw_data(filenames)

    class LemmaTokenizer(object):
        def __init__(self):
            self.wnl = WordNetLemmatizer()
        def __call__(self, doc):
            return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

    vectorizer = CountVectorizer(analyzer="word", tokenizer=LemmaTokenizer(),
                                 preprocessor=None, stop_words=util.stops,
                                 max_features=5000)

    train = [x[0] for x in data]
    # train = pickle.load(open(input('training data name: '), "rb"))
    # train = np.array(data)
    # trainx = train[:, 0]
    # trainy = train[:, 1]
    for d in train:
        print(d, end="\n\n")
    features = vectorizer.fit_transform(train)
    print(vectorizer.vocabulary_)
    # print(len(vectorizer.vocabulary_))
    # print(features.toarray())
    for f in features.toarray():
        print(f, end="\n\n")
    # print(vectorizer.get_stop_words())

def reformatdata():
    data = util.extract_raw_data(filenames, cap=8)

    lemmatizer = WordNetLemmatizer()
    with open("dataset/data/data.out", "w") as f:
        for d in data:
            words = word_tokenize()
            words = [lemmatizer.lemmatize(w) for w in words if w not in util.stops]
            sent = " ".join(words)
            f.write("{0}\n".format(sent))



if __name__ == '__main__':
    # vectorizeexample()
    reformatdata()

