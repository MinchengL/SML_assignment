from sklearn import model_selection, preprocessing, linear_model
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score
import pandas
import cleaning


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
            text = line.replace(label, "").lower()
            feature_list.append(text)
            counter = counter + 1
            if counter == 1000:
                break
    clean_feature_list = cleaning.clean_text(feature_list)
    data['label'] = label_list
    data['text'] = clean_feature_list
    return data


def logistics_regression(data):
    train_data, test_data, train_label, test_label = model_selection.train_test_split(data['text'], data['label'])
    label_encoder = preprocessing.LabelEncoder()
    encoded_train_label = label_encoder.fit_transform(train_label)
    encoded_test_label = label_encoder.fit_transform(test_label)

    tfidf_vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', max_df=0.5, max_features=5000)
    tfidf_vectorizer.fit(data['text'])
    tfidf_train_data = tfidf_vectorizer.transform(train_data)
    tfidf_test_data = tfidf_vectorizer.transform(test_data)

    logistics_regression_model = linear_model.LogisticRegression()
    #logistics_regression_model = linear_model.LogisticRegression(penalty='l2', dual=False,tol=0.0001, C=1.0,
    #                                                             fit_intercept=True, intercept_scaling=1,
    #                                                             solver='sag', max_iter=100, multi_class='ovr',
    #                                                             verbose=0, warm_start=False, n_jobs=1)
    logistics_regression_model.fit(tfidf_train_data, encoded_train_label)
    results = logistics_regression_model.predict(tfidf_test_data)
    accuracy = accuracy_score(results, encoded_test_label)
    print(accuracy)


if __name__ == '__main__':
    data =load_dataset("resource/train_tweets.txt")
    logistics_regression(data)