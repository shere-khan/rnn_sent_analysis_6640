import os, re, pickle, sys
from nltk.data import load
from nltk.corpus import stopwords
from bs4 import BeautifulSoup


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
    review = BeautifulSoup(review).get_text()
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

    # Process test data
    print("Processing test data...")
    testneg = '/home/justin/pycharmprojects/rnn_sent_analysis_6640/aclImdb/test/neg'
    test_neg_files = get_files_in_dir(testneg)
    testdata = processfiles(test_neg_files[:cap], 0)

    testpos = '/home/justin/pycharmprojects/rnn_sent_analysis_6640/aclImdb/test/pos'
    test_pos_files = get_files_in_dir(testpos)
    testdata.extend(processfiles(test_pos_files[:cap], 1))

    tokr = load('tokenizers/punkt/english.pickle')

    testsentences = list()
    for review in testdata:
        testsentences += getsentences(review, tokr)

    print("Creating test data pickle")
    pickle.dump(testsentences, open(input("Enter pickle name for train: "), "wb"))

    # Process training data
    print("Processing training data...")
    trainneg = '/home/justin/pycharmprojects/rnn_sent_analysis_6640/aclImdb/train/neg'
    train_neg_files = get_files_in_dir(trainneg)
    traindata = processfiles(train_neg_files[:cap], 0)

    trainpos = '/home/justin/pycharmprojects/rnn_sent_analysis_6640/aclImdb/train/pos'
    train_pos_files = get_files_in_dir(trainpos)
    traindata.extend(processfiles(train_pos_files[:cap], 1))

    sentences = list()
    for review in traindata:
        sentences += getsentences(review, tokr)

    print("Creating training data pickle")
    pickle.dump(sentences, open(input("Enter pickle name for train: "), "wb"))

    return sentences
