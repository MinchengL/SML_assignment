from sklearn import model_selection, preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import pandas


def knn_mothoed(data):
    train_data, test_data, train_label, test_label = model_selection.train_test_split(data['text'], data['label'])
    label_encoder = preprocessing.LabelEncoder()
    encoded_train_label = label_encoder.fit_transform(train_label)
    encoded_test_label = label_encoder.fit_transform(test_label)
    print("encoded label")

    tfidf_vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', ngram_range=(2, 3),
                                       max_df=0.5, max_features=15000)
    tfidf_vectorizer.fit(train_data.values.astype('U'))
    train_data = tfidf_vectorizer.transform(train_data.values.astype('U')).toarray()
    test_data = tfidf_vectorizer.transform(test_data.values.astype('U')).toarray()
    print("vectorized data")

    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(train_data, encoded_train_label)
    print("trained model")
    results = knn_model.predict(test_data)
    accuracy = accuracy_score(encoded_test_label, results)
    print(accuracy)
    print("predicted")


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
    data['text'] = feature_list
    data['label'] = label_list
    data = pandas.concat([data, data], axis=0)

    """
    cleaned_data_1 = pandas.read_csv('cleaned_data_2.csv', low_memory=False)
    csv_data_1 = pandas.DataFrame(cleaned_data_1)
    cleaned_data_2 = pandas.read_csv('cleaned_data_4.csv', low_memory=False)
    csv_data_2 = pandas.DataFrame(cleaned_data_2)
    cleaned_data = pandas.concat([csv_data_1, csv_data_2], axis=0)
    data = pandas.concat([cleaned_data, cleaned_data], axis=0)
    """

    print("loaded data")
    knn_mothoed(data)
