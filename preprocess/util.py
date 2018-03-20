import os, re, pickle, random, numpy as np
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
                data.append((line, [label]))

    return data

def getwords(review, remove_sw=False):
    review = BeautifulSoup(review, "html5lib").get_text()
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
            sents += getwords(raw, remove_sw)

    return sents, review[1]

def combinedata(neg, pos):
    neg.extend(pos)
    random.shuffle(neg)

    return neg

def extractdata(fn, cap, tokr, label):
    print("Processing data...")
    files = get_files_in_dir(fn)
    data = processfiles(files[:cap], label)

    sents = list()
    for one_line_one_review in data:
        sents.append(getsentences(one_line_one_review, tokr))

    return sents

def extract_sentences_and_flatten(reviews):
    sents = []
    for review in reviews:
        for r in review:
            sents.append(r[0])

    return sents

def get_len_longest_sentence(reviews):
    longest = 0
    for rev in reviews:
        length = len(rev[0])
        if length > longest:
            print(length)
            longest = length
            # longest = max(longest, len(rev[0]))

    return longest

def transform_reviews_to_word_vectors(reviews, model):
    longest = 0
    data = []
    labels = []
    vocab = set(model.wv.vocab)
    for rev in reviews:
        data1 = [model[d] for d in rev[0] if d in vocab]
        longest = max(longest, len(data1))
        labels.append(rev[1])
        data.append(np.array(data1))

    return np.array(data), np.array(labels), longest

def create_review_avgs_and_labels(reviews, model):
    vocab = set(model.wv.vocab)
    print('vocab len: ', len(vocab))
    features = list()
    labels = list()
    length = 0
    for rev in reviews:
        fvec = np.zeros(model.vector_size, dtype=np.float32)
        wordct = 0
        for w in rev[0]:
            if w in vocab:
                wordct += 1
                fvec = np.add(fvec, model[w])
        fvec = np.divide(fvec, wordct)
        features.append(fvec)
        size = len(fvec)
        if size > length:
            print('size fvec: ', size)
            length = size
        labels.append(rev[1])

    return np.array(features), np.array(labels)

def split_sents_and_labels(reviews):
    sents = [review[0] for review in reviews]
    labels = [review[1] for review in reviews]

    return sents, labels

def prepreprocessdata(cap=1000):
    # todo stpl size of trianing set test set
    # todo note # of pos reviews for train and test is equal
    tokr = load('tokenizers/punkt/english.pickle')

    if input("Process training data (y/n)?") == 'y':
        # Process training data
        print("Processing training data...")
        trainneg = '/home/justin/pycharmprojects/rnn_sent_analysis_6640/aclImdb/train/neg'
        train_neg_sents = extractdata(trainneg, cap, tokr, 0)

        trainpos = '/home/justin/pycharmprojects/rnn_sent_analysis_6640/aclImdb/train/pos'
        train_pos_sents = extractdata(trainpos, cap, tokr, 1)

        labled_train_reviews = combinedata(train_neg_sents, train_pos_sents)

        print("Creating pickle for processed data")
        pickle.dump(labled_train_reviews, open(input("Enter pickle name: "), "wb"))

    if input("Process test data (y/n)?") == 'y':
        # Process test data
        print("Processing test data...")
        testneg = '/home/justin/pycharmprojects/rnn_sent_analysis_6640/aclImdb/test/neg'
        test_neg_sents = extractdata(testneg, cap, tokr, 0)
        testpos = '/home/justin/pycharmprojects/rnn_sent_analysis_6640/aclImdb/test/pos'
        test_pos_sents = extractdata(testpos, cap, tokr, 1)

        labled_test_reviews = combinedata(test_neg_sents, test_pos_sents)

        print("Creating pickle for processed data")
        pickle.dump(labled_test_reviews, open(input("Enter pickle name: "), "wb"))
