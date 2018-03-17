import os, re
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer


stops = set(stopwords.words('english'))

def get_files_in_dir(dir):
    files = list()
    for file in os.listdir(dir):
        if file.endswith(".txt"):
            files.append(os.path.join(dir, file))

    return files

def processfiles(files):
    data = list()
    for file in files:
        with open(file) as f:
            for line in f:
                review = BeautifulSoup(line).get_text()
                review = remove_emoji_and_nums(review)
                words = review.lower().split()
                words = [w for w in words if w not in stops]
                data.append(" ".join(words))

    return data

def remove_emoji_and_nums(text):
    emojis = """:-) :) :o) :] :3 :c) :> =] 8) =) :} :^) 
    :D 8-D 8D x-D xD X-D XD =-D =D =-3 =3 B^D :( :/ :-( :'( :D :P""".split()
    emojipat = "|".join(map(re.escape, emojis))
    text = re.sub("[^a-zA-Z0-9{0}]".format(emojipat), " ", text)

    return text

if __name__ == '__main__':
    testneg = '/home/justin/pycharmprojects/rnn_sent_analysis_6640/aclImdb/test/neg'
    testpos = '/home/justin/pycharmprojects/rnn_sent_analysis_6640/aclImdb/test/pos'
    trainneg = '/home/justin/pycharmprojects/rnn_sent_analysis_6640/aclImdb/train/neg'
    trainpos = '/home/justin/pycharmprojects/rnn_sent_analysis_6640/aclImdb/train/pos'

    test_neg_files = get_files_in_dir(testneg)
    test_neg_proc_reviews = processfiles(test_neg_files)


    # test_pos_files = get_files_in_dir(testpos)
    # test_pos_proc_reviews = processfiles(test_pos_files)
    #
    # train_neg_files = get_files_in_dir(trainneg)
    # test_pos_proc_reviews = processfiles(train_neg_files)
    #
    # train_pos_files = get_files_in_dir(trainpos)
    # test_pos_proc_reviews = processfiles(train_pos_files)

    # stpl: something to play with
    # todo stpl max_features
    vectorizer = CountVectorizer(analyzer='word',
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=5000)
