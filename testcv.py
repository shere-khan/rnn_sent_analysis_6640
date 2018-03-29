import util, pickle, numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

filenames = ['/home/justin/pycharmprojects/rnn_sent_analysis_6640/reviews/train/neg/',
             '/home/justin/pycharmprojects/rnn_sent_analysis_6640/reviews/train/pos/',
             '/home/justin/pycharmprojects/rnn_sent_analysis_6640/reviews/test/neg/',
             '/home/justin/pycharmprojects/rnn_sent_analysis_6640/reviews/test/pos/']

def vectorizeexample():
    data = util.extract_raw_data(['/home/justin/pycharmprojects/rnn_sent_analysis_6640'
                                  '/data/processed/'])

    # class LemmaTokenizer(object):
    #     def __init__(self):
    #         self.wnl = WordNetLemmatizer()
    #     def __call__(self, doc):
    #         return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

    # vectorizer = CountVectorizer(analyzer="word", tokenizer=LemmaTokenizer(),
    #                              preprocessor=None, stop_words=util.stops,
    #                              max_features=1000)

    vectorizer = CountVectorizer(analyzer="word", preprocessor=None, max_features=1000)

    train = [x[0] for x in data]

    # train = pickle.load(open(input('training data name: '), "rb"))
    # train = np.array(data)
    # trainx = train[:, 0]
    # trainy = train[:, 1]
    # for d in train:
    #     print(d, end="\n\n")
    # print(len(vectorizer.vocabulary_))
    # print(features.toarray())
    # for f in features.toarray():
    #     print(f, end="\n\n")
    # print(vectorizer.get_stop_words())

    vectorizer.fit_transform(train)
    print(vectorizer.vocabulary_)
    pickle.dump(vectorizer, open("data/bow_mod1000", "wb"))


def reformatdata(stops=False):
    data = util.extract_raw_data(filenames, cap=None)

    lemmatizer = WordNetLemmatizer()
    with open("data/data1.out", "w") as f:
        for d in data:
            review = BeautifulSoup(d[0], "html5lib").get_text()
            review = util.remove_emoji_and_nums(review)
            words = word_tokenize(review)
            if stops:
                words = [lemmatizer.lemmatize(w) for w in words if w not in util.stops]
            sent = " ".join(words)
            f.write("{0}\n".format(sent))



if __name__ == '__main__':
    vectorizeexample()
    # reformatdata(True)

