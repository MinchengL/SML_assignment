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
            if line is not None or lien is not "":
                label = line.split('\t')[0]
                text = line.replace(label, "")
                labels.append(label)
                texts.append(text)
                
    with codecs.open('train_data.txt', 'w', encoding='utf-8') as train_file:
        for i in range(len(texts)):
            train_file.write("__label__"+str(labels[i])+","+str(texts[i]))

def classify():
    load_data()
    test_text = []
    model = fasttext.train_supervised('train_data.txt',  epoch=20, minn=4, maxn=7, lr=0.8, loss='hs', wordNgrams=3, dim=200)
    with codecs.open('cleaned_unlabel_data.txt', 'r', encoding='utf-8') as test_file:
        for line in test_file.readlines():
            test_text.append(line)
    results = []
    for i in range(len(test_text)):
        result = model.predict(test_text[i].replace('\n',''))
        results.append(result[0][0].replace('__label__','').replace(',',''))
    results_df = pandas.DataFrame(results)
    results_df.to_csv('result_fasttext.csv')
    print("finished")

if __name__ == '__main__':
    classify()
