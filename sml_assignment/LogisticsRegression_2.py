from sklearn import model_selection, preprocessing, linear_model
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import LoadDataSet
import pandas, scipy
import numpy as np
from sklearn.metrics import accuracy_score


def logistics_regression(data):
    label_list = list(set(data['label']))
    results = []
    for i in range(len(label_list)):

        train_data, test_data, train_label, test_label = model_selection.train_test_split(data['text'], data['label'])
        train_label = train_label.tolist()
        train_relabel = []
        for j in range(len(train_label)):
            if train_label[j] == label_list[i]:
                train_relabel.append(1)
            else:
                train_relabel.append(0)
        label_encoder = preprocessing.LabelEncoder()
        encoded_train_relabel = label_encoder.fit_transform(train_relabel)
        print(1 in encoded_train_relabel)

        tfidf_vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', max_df=0.3)
        tfidf_vectorizer.fit(data['text'])
        tfidf_train_data = tfidf_vectorizer.transform(train_data)
        tfidf_test_data = tfidf_vectorizer.transform(test_data)

        logistics_regression_model = linear_model.LogisticRegression()
        logistics_regression_model.fit(tfidf_train_data, encoded_train_relabel)
        single_result = logistics_regression_model.predict_proba(tfidf_test_data)
        result = []
        for item in single_result:
            result.append(item[1])
        results.append(result)
    result_matrix = np.mat(results).T.A
    result_index = result_matrix.argmax(axis=1)
    final_test_results = []
    for i in result_index:
        final_test_results.append(label_list[i])
    accuracy = accuracy_score(test_label, final_test_results)
    print(accuracy)


if __name__ == '__main__':
    data = LoadDataSet.load_dataset("resource/train_tweets.txt")
    logistics_regression(data)