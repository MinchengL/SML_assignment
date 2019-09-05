import pandas


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
            if counter == 10000:
                break
    data['label'] = label_list
    data['text'] = feature_list
    return data
