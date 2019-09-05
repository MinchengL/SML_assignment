import FeatureExtraction


def load_dataset(filename):
    label_list = []
    feature_list = []
    counter = 0
    with open(filename, 'r') as file:
        for line in file.readlines():
            results = line.split("	")
            label = results[0]
            label_list.append(int(label))
            text = line.replace(label, "").strip()
            feature = FeatureExtraction.feature_extraction([text])
            feature_list.append(feature)
            counter = counter + 1
            if counter == 10000:
                break
    return label_list, feature_list


def load_test_dataset(filename):
    test_label_list = []
    test_feature_list = []
    counter = 0
    with open(filename, 'r') as file:
        for line in file.readlines():
            results = line.split("	")
            label = results[0]
            test_label_list.append(int(label))
            text = line.replace(label, "").strip()
            feature = FeatureExtraction.feature_extraction([text])
            test_feature_list.append(feature)
            counter = counter + 1
            if counter == 10000:
                break
    return test_label_list, test_feature_list
