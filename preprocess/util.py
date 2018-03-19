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

def combinedata(neg, pos):
    reviews = neg.extend(pos)
    reviews.shuffle()

    return reviews

def changedata(fn, cap, tokr, label):
    print("Processing data...")
    files = get_files_in_dir(fn)
    data = processfiles(files[:cap], label)

    sents = list()
    for review in data:
        sents += getsentences(review, tokr)

    return sents

def prepreprocessdata(cap=1000):
    # todo stpl size of trianing set test set
    # todo note # of pos reviews for train and test is equal
    tokr = load('tokenizers/punkt/english.pickle')

    # Process test data
    testneg = '/home/justin/pycharmprojects/rnn_sent_analysis_6640/aclImdb/test/neg'
    test_neg_sents = changedata(testneg, cap, tokr, 0)

    testpos = '/home/justin/pycharmprojects/rnn_sent_analysis_6640/aclImdb/test/pos'
    test_pos_sents = changedata(testpos, cap, tokr, 1)

    labled_train_reviews = combinedata(test_neg_sents, test_pos_sents)

    print("Creating pickle for processed data")
    pickle.dump(labled_train_reviews, open(input("Enter pickle name: "), "wb"))


    # Process training data
    print("Processing training data...")
    trainneg = '/home/justin/pycharmprojects/rnn_sent_analysis_6640/aclImdb/train/neg'
    train_neg_sents = changedata(trainneg, cap, tokr, 0)

    trainpos = '/home/justin/pycharmprojects/rnn_sent_analysis_6640/aclImdb/train/pos'
    train_pos_sents = changedata(trainpos, cap, tokr, 1)

    labled_test_reviews = combinedata(train_neg_sents, train_pos_sents)

    print("Creating pickle for processed data")
    pickle.dump(labled_test_reviews, open(input("Enter pickle name: "), "wb"))

