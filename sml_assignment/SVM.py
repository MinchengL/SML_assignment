from sklearn import model_selection, preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import pandas


def svm(data):

    train_data, test_data, train_label, test_label = model_selection.train_test_split(data['text'], data['label'])
    label_encoder = preprocessing.LabelEncoder()
    encoded_train_label = label_encoder.fit_transform(train_label)
    encoded_test_label = label_encoder.fit_transform(test_label)
    print("encoded label")

    tfidf_vectorizer = CountVectorizer(stop_words='english',ngram_range=(1,3))
    tfidf_vectorizer.fit(train_data.values.astype('U'))
    tfidf_train_data = tfidf_vectorizer.transform(train_data.values.astype('U'))
    tfidf_test_data = tfidf_vectorizer.transform(test_data.values.astype('U'))
    print("vectorized data")

    normalizer = preprocessing.Normalizer()
    normalizer.fit(tfidf_train_data)
    normalizer.fit(tfidf_test_data)
    normalizer.transform(tfidf_train_data)
    normalizer.transform(tfidf_test_data)
    print("normalized data")

    svm_model = LinearSVC()
    svm_model.fit(tfidf_train_data, encoded_train_label)
    results = svm_model.predict(tfidf_test_data)
    print("predicted")
    accuracy = accuracy_score(encoded_test_label, results)
    print(accuracy)


if __name__ == '__main__':
    """
    cleaned_data_1 = pandas.read_csv('cleaned_data_2.csv', low_memory=False)
    csv_data_1 = pandas.DataFrame(cleaned_data_1)
    cleaned_data_2 = pandas.read_csv('cleaned_data_4.csv', low_memory=False)
    csv_data_2 = pandas.DataFrame(cleaned_data_2)
    cleaned_data = pandas.concat([csv_data_1, csv_data_2], axis=0)
    """

    label_list = []
    feature_list = []
    data = pandas.DataFrame()
    counter = 0
    #with open('cleaned_data.txt', 'r', encoding='utf-8') as file:
    with open('train_tweets.txt', 'r', encoding='utf-8') as file:
        for line in file.readlines():
            if counter < 1000:
                results = line.split("	")
                label = results[0]
                label_list.append(int(label))
                text = line.replace(label, "").lower()
                feature_list.append(text)
            counter = counter + 1

            # clean_feature_list = cleaning.clean_text(feature_list)
    data['label'] = label_list
    data['text'] = feature_list


    data = pandas.concat([data[1:10000],data[1:10000]], axis=0)
    print("loaded data")
    svm(data)