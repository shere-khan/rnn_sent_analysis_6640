from nltk.corpus import stopwords

if __name__ == '__main__':
    stops = set(stopwords.words('english'))
    words = [w for w in words if w not in stops]