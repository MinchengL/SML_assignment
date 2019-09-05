from sklearn import model_selection, preprocessing, linear_model
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import LoadDataSet
import pandas, scipy
from sklearn.metrics import accuracy_score


def logistics_regression(data):
    train_data, test_data, train_label, test_label = model_selection.train_test_split(data['text'], data['label'])
    label_encoder = preprocessing.LabelEncoder()
    encoded_train_label = label_encoder.fit_transform(train_label)
    encoded_test_label = label_encoder.fit_transform(test_label)

    tfidf_vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', token_pattern=r'\w{1,}')
    tfidf_vectorizer.fit(data['text'])
    tfidf_train_data = tfidf_vectorizer.transform(train_data)
    tfidf_test_data = tfidf_vectorizer.transform(test_data)
    count_vectorizer = CountVectorizer(analyzer='word', stop_words='english', token_pattern=r'\w{1,}')
    count_vectorizer.fit(data['text'])
    count_train_data = count_vectorizer.transform(train_data)
    count_test_data = count_vectorizer.transform(test_data)
    final_train_data = scipy.sparse.hstack([tfidf_train_data, count_train_data])
    final_test_data = scipy.sparse.hstack([tfidf_test_data, count_test_data])

    logistics_regression_model = linear_model.LogisticRegression()
    logistics_regression_model.fit(final_train_data, encoded_train_label)
    results = logistics_regression_model.predict(final_test_data)
    accuracy = accuracy_score(results, encoded_test_label)
    print(accuracy)


if __name__ == '__main__':
    data = LoadDataSet.load_dataset("resource/train_tweets.txt")
    logistics_regression(data)