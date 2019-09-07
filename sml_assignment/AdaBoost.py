from sklearn import model_selection, preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
import pandas
import cleaning


def load_dataset(filename):
    label_list = []
    feature_list = []
    data = pandas.DataFrame()
    counter = 0
    with open(filename, 'r') as file:
        for line in file.readlines():
            results = line.split("	")
            label = results[0]
            label_list.append(int(label))
            text = line.replace(label, "").lower()
            feature_list.append(text)
            counter = counter + 1
            if counter == 100000:
                break
    clean_feature_list = cleaning.clean_text(feature_list)
    data['label'] = label_list
    data['text'] = clean_feature_list
    return data


def adaboost(data):
    label_encoder = preprocessing.LabelEncoder()
    label_num = len(list(set(data['label'])))
    label = label_encoder.fit_transform(data['label'])

    tfidf_vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', max_df=0.3)
    tfidf_vectorizer.fit(data['text'])
    data = tfidf_vectorizer.transform(data['text'])

    adaboost_model = AdaBoostClassifier(n_estimators=label_num)
    score = model_selection.cross_val_score(adaboost_model, data, label)
    print(score.mean())


if __name__ == '__main__':
    data = load_dataset("resource/train_tweets.txt")
    adaboost(data)