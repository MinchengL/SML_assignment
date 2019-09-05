from sklearn.feature_extraction.text import CountVectorizer
import string
import re


def vectorizer_initailizer(text):
    try:
        count_vectorizer = CountVectorizer(stop_words=None, analyzer='word', ngram_range=(1,1))
        verctorizer_fit = count_vectorizer.fit(text)
        return verctorizer_fit
    except ValueError as e:
        return None


def get_words_num(vectorizer):
    words_num = len(vectorizer.vocabulary_)
    return words_num


def get_sentence_length(vectorizer):
    sentence_length = 0

    for i in vectorizer.vocabulary_:
        sentence_length = sentence_length + vectorizer.vocabulary_[i]
    return sentence_length


def get_punctuation_num(text):
    punctuation_num = 0
    for i in text[0]:
        if i in string.punctuation:
            punctuation_num = punctuation_num + 1
    return punctuation_num


def get_emoji_num(text):
    emoji_num = 0
    emoji_pattern = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')
    result = re.findall(emoji_pattern, text)
    emoji_num = len(result)
    return emoji_num


def feature_extraction(text):
    features_matrix = []
    count_verctorizer = vectorizer_initailizer(text)
    if count_verctorizer is not None:
        features_matrix.append(get_sentence_length(count_verctorizer))
        features_matrix.append(get_words_num(count_verctorizer))
        features_matrix.append(get_punctuation_num(text))
    return features_matrix
