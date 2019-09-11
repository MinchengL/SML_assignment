from sklearn import model_selection, preprocessing, linear_model
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
import pandas


def logistics_regression(data):
    parameters = [{'penalty': ['l1', 'l2'],
                         'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100],
                         'solver': ['liblinear'],
                         'multi_class': ['ovr']},
                        {'penalty': ['l2'],
                         'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100],
                         'solver': ['lbfgs'],
                         'multi_class': ['ovr', 'multinomial']}]

    train_data, test_data, train_label, test_label = model_selection.train_test_split(data['text'], data['label'])
    label_encoder = preprocessing.LabelEncoder()
    encoded_train_label = label_encoder.fit_transform(train_label)
    encoded_test_label = label_encoder.fit_transform(test_label)

    tfidf_vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', max_df=0.38, max_features=15000)
    tfidf_vectorizer.fit(train_data.values.astype('U'))
    train_data = tfidf_vectorizer.transform(train_data.values.astype('U'))
    test_data = tfidf_vectorizer.transform(test_data.values.astype('U'))

    normalizer = preprocessing.Normalizer()
    normalizer.fit(train_data)
    normalizer.transform(train_data)
    normalizer.transform(test_data)

    logistics_regression_model = GridSearchCV(linear_model.LogisticRegression(), parameters, cv=5, scoring='accuracy')
    logistics_regression_model.fit(train_data, encoded_train_label)
    results = logistics_regression_model.predict(test_data)
    accuracy = accuracy_score(results, encoded_test_label)
    print(accuracy)
    print("predicted")


if __name__ == '__main__':

    label_list = []
    feature_list = []
    data = pandas.DataFrame()
    counter = 0
    with open('cleaned_data.txt', 'r', encoding='utf-8') as file:
        for line in file.readlines():
            results = line.split("	")
            label = results[0]
            label_list.append(int(label))
            text = line.replace(label, "").lower()
            feature_list.append(text)
    data['text'] = feature_list
    data['label'] = label_list
    data = pandas.concat([data, data], axis=0)
    """
    cleaned_data_1 = pandas.read_csv('cleaned_data_2.csv', low_memory=False)
    csv_data_1 = pandas.DataFrame(cleaned_data_1)
    cleaned_data_2 = pandas.read_csv('cleaned_data_4.csv', low_memory=False)
    csv_data_2 = pandas.DataFrame(cleaned_data_2)
    cleaned_data = pandas.concat([csv_data_1, csv_data_2], axis=0)
    data = pandas.concat([cleaned_data[0:10000], cleaned_data[0:10000]], axis=0)
    """

    print("loaded data")
    logistics_regression(data)
