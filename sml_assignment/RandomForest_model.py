from sklearn import model_selection, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
import pandas, numpy


def randomforest_model(data):

    train_data, test_data, train_label, test_label = model_selection.train_test_split(data['text'], data['label'])
    label_encoder = preprocessing.LabelEncoder()
    encoded_train_label = label_encoder.fit_transform(train_label)
    encoded_test_label = label_encoder.fit_transform(test_label)
    print("encoded label")

    tfidf_vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', max_df=0.5, max_features=15000,
                                       ngram_range=(1, 3))
    tfidf_vectorizer.fit(train_data.values.astype('U'))
    tfidf_vectorizer.fit(test_data.values.astype('U'))
    tfidf_train_data = tfidf_vectorizer.transform(train_data.values.astype('U'))
    tfidf_test_data = tfidf_vectorizer.transform(test_data.values.astype('U'))
    print("vectorized data")

    normalizer = preprocessing.Normalizer()
    normalizer.fit(tfidf_train_data)
    normalizer.fit(tfidf_test_data)
    normalizer.transform(tfidf_train_data)
    normalizer.transform(tfidf_test_data)
    print("normalized data")

    random_paras = {'n_estimators': [int(x) for x in numpy.linspace(start=10, stop=200, num = 10)],
                   'max_features': ['auto', 'log2'],
                   'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
                   'min_samples_split': [2, 5, 8, 19],
                   'min_samples_leaf': [1, 2, 4],
                   'bootstrap': [True, False]}

    knn_model = RandomForestClassifier()
    rf_random_search = RandomizedSearchCV(estimator=knn_model, param_distributions=random_paras, n_iter=100, cv=5)
    rf_random_search.fit(tfidf_train_data, encoded_train_label)
    print("trained model")
    best_paras = rf_random_search.best_params_
    print(best_paras)
    knn_model = RandomForestClassifier(n_estimators=best_paras['n_estimators'],
                                        max_features=best_paras['max_features'],
                                        max_depth=best_paras['max_depth'],
                                        min_samples_split=best_paras['min_samples_split'],
                                        min_samples_leaf=best_paras['min_samples_leaf'],
                                        bootstrap=best_paras['bootstrap'])
    results = knn_model.predict(tfidf_test_data)
    print("predicted")
    accuracy = accuracy_score(results, encoded_test_label)
    print(accuracy)


if __name__ == '__main__':
    label_list = []
    feature_list = []
    data = pandas.DataFrame()
    counter = 0
    with open('train_tweets.txt', 'r', encoding='utf-8') as file:
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
    data = pandas.concat([cleaned_data[0:10000],cleaned_data[0:10000]], axis=0)
    """
    print("loaded data")
    randomforest_model(data)
