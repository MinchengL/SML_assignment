from sklearn import model_selection, preprocessing, linear_model, dummy, ensemble
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import LoadDataSet
import pandas, scipy
from sklearn.metrics import accuracy_score


def dummy_classifier(data):
    train_data, test_data, train_label, test_label = model_selection.train_test_split(data['text'], data['label'])
    label_encoder = preprocessing.LabelEncoder()
    encoded_train_label = label_encoder.fit_transform(train_label)
    encoded_test_label = label_encoder.fit_transform(test_label)

    tfidf_vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', max_df=0.3)
    tfidf_vectorizer.fit(data['text'])
    tfidf_train_data = tfidf_vectorizer.transform(train_data)
    tfidf_test_data = tfidf_vectorizer.transform(test_data)
    count_vectorizer = CountVectorizer(analyzer='word', stop_words='english', max_df=0.3)
    count_vectorizer.fit(data['text'])
    count_train_data = count_vectorizer.transform(train_data)
    count_test_data = count_vectorizer.transform(test_data)
    final_train_data = scipy.sparse.hstack([tfidf_train_data, count_train_data])
    final_test_data = scipy.sparse.hstack([tfidf_test_data, count_test_data])

    model = ensemble.GradientBoostingClassifier(n_estimators=100)
    model.fit(final_train_data, encoded_train_label)
    results = model.predict(final_test_data)
    accuracy = accuracy_score(results, encoded_test_label)
    print(accuracy)


if __name__ == '__main__':
    data = LoadDataSet.load_dataset("resource/train_tweets.txt")
    dummy_classifier(data)