import os, re, pickle, sys
import logging
from nltk.data import load
from nltk.corpus import stopwords
from nltk import tokenize
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec


stops = set(stopwords.words('english'))

def get_files_in_dir(dir):
    files = list()
    for file in os.listdir(dir):
        if file.endswith(".txt"):
            files.append(os.path.join(dir, file))

    return files

def remove_emoji_and_nums(text):
    emojis = """:-) :) :o) :] :3 :c) :> =] 8) =) :} :^) 
    :D 8-D 8D x-D xD X-D XD =-D =D =-3 =3 B^D :( :/ :-( :'( :D :P""".split()
    emojipat = "|".join(map(re.escape, emojis))
    text = re.sub("[^a-zA-Z0-9{0}]".format(emojipat), " ", text)

    return text

def processfiles(files, label):
    data = list()
    for file in files:
        with open(file) as f:
            for line in f:
                data.append((line, label))

    return data

def getwords(review, remove_sw=False):
    review = BeautifulSoup(review, 'lxml').get_text()
    review = remove_emoji_and_nums(review)
    words = review.lower().split()
    if remove_sw:
        words = [w for w in words if w not in stops]

    return words

def getsentences(review, tokenizer, remove_sw=False):
    raws = tokenizer.tokenize(review[0].strip())
    sents = list()
    for raw in raws:
        if len(raw) > 0:
            sents.append(getwords(raw, remove_sw))

    return sents

def prepreprocessdata():
    # todo stpl size of trianing set test set
    # todo note # of pos reviews for train and test is equal
    cap = -1

    # testneg = '/home/justin/pycharmprojects/rnn_sent_analysis_6640/aclImdb/test/neg'
    # test_neg_files = get_files_in_dir(testneg)
    # testdata = processfiles(test_neg_files[:cap], 0)

    # testpos = '/home/justin/pycharmprojects/rnn_sent_analysis_6640/aclImdb/test/pos'
    # test_pos_files = get_files_in_dir(testpos)
    # testdata.extend(processfiles(test_pos_files[:cap], 1))

    trainneg = '/home/justin/pycharmprojects/rnn_sent_analysis_6640/aclImdb/train/neg'
    train_neg_files = get_files_in_dir(trainneg)
    traindata = processfiles(train_neg_files[:cap], 0)

    trainpos = '/home/justin/pycharmprojects/rnn_sent_analysis_6640/aclImdb/train/pos'
    train_pos_files = get_files_in_dir(trainpos)
    traindata.extend(processfiles(train_pos_files[:cap], 1))

    # stpl: something to play with
    # todo stpl max_features
    # vectorizer = CountVectorizer(analyzer='word',
    #                              tokenizer=None,
    #                              preprocessor=None,
    #                              stop_words=None)

    tokr = load('tokenizers/punkt/english.pickle')

    sentences = list()
    for review in traindata:
        sentences += getsentences(review, tokr)

    return sentences

if __name__ == '__main__':
    isfilesaved = False
    f = open(sys.argv[1])
    isfilesaved = True if f else False
    if not isfilesaved:
        sentences = prepreprocessdata()
        pickle.dump(sentences, open("sentences.p", "wb"))
    else:
        sentences = pickle.load(open("sentences.p", "rb"))
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

    num_features = 300  # Word vector dimensionality
    min_word_count = 40  # Minimum word count
    num_workers = 4  # Number of threads to run in parallel
    context = 10  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words

    print("W2V training...")
    model = Word2Vec(sentences, workers=num_workers, size=num_features,
                     min_count=min_word_count, window=context, sample=downsampling,
                     seed=1)

    # If no further training, makes more mem efficient
    model.init_sims(replace=True)

    # Save the model
    model.save("word2vecmodel")
