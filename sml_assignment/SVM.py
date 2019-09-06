from sklearn import model_selection, preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
import pandas, scipy
from sklearn.metrics import accuracy_score


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
            text = line.replace(label, "")
            feature_list.append(text)
            counter = counter + 1
            if counter == 100000:
                break
    data['label'] = label_list
    data['text'] = feature_list
    return data


def svm(data):
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

    svm_model = LinearSVC()
    svm_model.fit(final_train_data, encoded_train_label)
    results = svm_model.predict(final_test_data)
    accuracy = accuracy_score(results, encoded_test_label)
    print(accuracy)


if __name__ == '__main__':
    data = load_dataset("resource/train_tweets.txt")
    svm(data)