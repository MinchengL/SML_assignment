import fasttext
import pandas
import codecs
from sklearn.model_selection import train_test_split
import numpy as np

def load_data():
    labels = []
    texts = []

    with codecs.open('cleaned_data.txt', 'r', encoding='utf-8') as file:
        for line in file.readlines():
            if line is not None or line is not "":
                label = line.split('\t')[0]
                text = line.replace(label, "")
                labels.append(label)
                texts.append(text)

    train_data, test_data, train_label, test_label = train_test_split(texts, labels)
    train_data = list(train_data)
    test_data = list(test_data)
    train_label = list(train_label)
    test_label = list(test_label)
    with codecs.open('train_data.txt', 'w', encoding='utf-8') as train_file:
        for i in range(len(train_data)):
            train_file.write("__label__"+str(train_label[i])+","+str(train_data[i]))
    return test_data, test_label

def classify():
    test_text, test_label = load_data()
    model = fasttext.train_supervised('train_data.txt', epoch=20, minn=5, maxn=8, lr=0.8, loss='hs', wordNgrams=3, dim=200)
    results = []
    counter = 0
    for i in range(len(test_text)):
        result = model.predict(test_text[i].replace('\n',''))
        results.append(result[0][0].replace('__label__','').replace(',',''))
        print(result[0][0].replace('__label__','').replace(',',''))
    for i in range(len(results)):
        if(results[i]==test_label[i]):
            counter = counter+1
    print(counter/len(results))

if __name__ == '__main__':
    classify()
