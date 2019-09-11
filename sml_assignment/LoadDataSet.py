import pandas
import cleaning
import AdaBoost
import LogisticImplement
import SVM
import codecs


def load_dataset(filename):
    label_list = []
    feature_list = []
    data = pandas.DataFrame()
    counter = 0
    with codecs.open(filename, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            if counter < 20000:
                results = line.split("	")
                label = results[0]
                label_list.append(int(label))
                text = line.replace(label, "").lower()
                feature_list.append(text)
            counter = counter + 1    
            
    #clean_feature_list = cleaning.clean_text(feature_list)
    data['label'] = label_list
    data['text'] = feature_list
    data.to_csv("cleaned_data_3.csv")
    return data


if __name__ == '__main__':
    data = load_dataset("train_tweets.txt")
    """
    print("the result of adaboost is ")
    AdaBoost.adaboost(data)
    print("the result of SVM is ")
    SVM.svm(data)
    print("the result of logistics is ")
    LogisticImplement.logistics_regression(data)
    print("the result of fasttext is ")
    fasttext_model.fasttext_classify()
    """
