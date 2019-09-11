from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
import pandas


def svm(data, test_data):

    train_data = data['text']
    train_label = data['label']
    test_data = test_data['test']
    label_encoder = preprocessing.LabelEncoder()
    encoded_train_label = label_encoder.fit_transform(train_label)
    print("encoded label")

    tfidf_vectorizer = CountVectorizer(analyzer='word', stop_words='english', max_features=30000)
    tfidf_vectorizer.fit(train_data.values.astype('U'))
    train_data = tfidf_vectorizer.transform(train_data.values.astype('U'))
    test_data = tfidf_vectorizer.transform(test_data.values.astype('U'))
    print("vectorized data")

    normalizer = preprocessing.Normalizer()
    normalizer.fit(train_data)
    normalizer.transform(train_data)
    normalizer.transform(test_data)
    print("normalized data")

    svm_model = LinearSVC()
    svm_model.fit(train_data, encoded_train_label)
    results = svm_model.predict(test_data)
    print("predicted")
    inversed_labels = label_encoder.inverse_transform(results)
    results_df = pandas.DataFrame(inversed_labels)
    results_df.to_csv("results_svm_3.csv")


if __name__ == '__main__':
    label_list = []
    feature_list = []
    data = pandas.DataFrame()
    counter = 0
    #with open('cleaned_data.txt', 'r', encoding='utf-8') as file:
    with open('train_tweets.txt', 'r', encoding='utf-8') as file:
        for line in file.readlines():
            results = line.split("	")
            label = results[0]
            label_list.append(int(label))
            text = line.replace(label, "").lower()
            feature_list.append(text)
    data['text']=feature_list
    data['label']=label_list
    #data = pandas.concat([data,data], axis=0)

    test_csv_data = []
    #with open('cleaned_unlabel_data.txt', 'r', encoding='utf-8') as file:
    with open('test_tweets_unlabeled.txt', 'r', encoding='utf-8') as file:
        for line in file.readlines():
            if line is not '\n':
                test_csv_data.append(line)

    test_data = pandas.DataFrame()
    test_data['test'] = test_csv_data
    print("loaded data")
    svm(data, test_data)
