import pandas
import cleaning
import AdaBoost
import LogisticImplement
import SVM


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
            if counter == 100000:
                break
    clean_feature_list = cleaning.clean_text(feature_list)
    data['label'] = label_list
    data['text'] = clean_feature_list
    return data


if __name__ == '__main__':
    data = load_dataset("resource/train_tweets.txt")
    print("the result of adaboost is ")
    AdaBoost.adaboost(data)
    print("the result of SVM is ")
    SVM.svm(data)
    print("the result of logistics is ")
    LogisticImplement.logistics_regression(data)
