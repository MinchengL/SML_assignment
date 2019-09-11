from nltk.stem import PorterStemmer
from textblob import TextBlob, Word
from nltk.stem import WordNetLemmatizer
import nltk
import re
from urlextract import URLExtract


def clean_text(file):
    data = file
    cleaned = []
    stemming = PorterStemmer()
    lemmatization = WordNetLemmatizer()
    extractor = URLExtract()
    counter=0

    for item in data:
        item = item.lower()
        for i in extractor.find_urls(item):
            item = item.replace(i, '')
        item = item.replace('handle', '')
        item = item.replace('rt', '')
        item = item.replace(r'\W', '')
        puncuation = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
        item = re.sub(pattern=puncuation, repl=' ', string=item)
        item = re.sub(
            u"(\ud83d[\ude00-\ude4f])|"  # emoticons
            u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
            u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
            u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
            u"(\ud83c[\udde0-\uddff])"  # flags (iOS)
            "+", flags=re.UNICODE, repl=" ", string=item)
        words = item.split()
        words = [Word(word).correct() for word in words]
        item = " ".join(words)
        #words = [stemming.stem(word=word) for word in words]
        #words = [lemmatization.lemmatize(word=word) for word in words]
        cleaned.append(item)
        print(item)
        print(counter)
        counter = counter+1

    return cleaned







