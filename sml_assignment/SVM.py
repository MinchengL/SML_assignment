from sklearn import model_selection, preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import LoadDataSet


def svm(data):
    train_data, test_data, train_label, test_label = model_selection.train_test_split(data['text'], data['label'])
    label_encoder = preprocessing.LabelEncoder()
    encoded_train_label = label_encoder.fit_transform(train_label)
    encoded_test_label = label_encoder.fit_transform(test_label)

    tfidf_vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', max_df=0.3)
    tfidf_vectorizer.fit(data['text'])
    tfidf_train_data = tfidf_vectorizer.transform(train_data)
    tfidf_test_data = tfidf_vectorizer.transform(test_data)

    svm_model = LinearSVC()
    svm_model.fit(tfidf_train_data, encoded_train_label)
    results = svm_model.predict(tfidf_test_data)
    accuracy = accuracy_score(results, encoded_test_label)
    print(accuracy)


if __name__ == '__main__':
    data = LoadDataSet.load_dataset("resource/train_tweets.txt")
    svm(data)