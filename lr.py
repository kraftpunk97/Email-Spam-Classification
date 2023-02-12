import os
import random
import warnings
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import binarize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# To deal with the overflow warning given by np.exp;
# because it's ugly and I just don't like it :(
warnings.filterwarnings('once')

stemmer = PorterStemmer()
analyzer = CountVectorizer().build_analyzer()
def stemmed_words(string): return (stemmer.stem(word) for word in analyzer(string))


def create_classes(file_name_list): return np.array([('spam' in file_name) for file_name in file_name_list],
                                                    dtype=np.int64)


def train_val_split(file_name_list):
    ham_fname_list = [file_name for file_name in file_name_list if 'ham' in file_name]
    spam_fname_list = [file_name for file_name in file_name_list if 'spam' in file_name]

    ham_train_set, ham_val_set = train_test_split(ham_fname_list, train_size=0.7)  # For ham
    spam_train_set, spam_val_set = train_test_split(spam_fname_list, train_size=0.7)  # For spam

    train_set = ham_train_set + spam_train_set
    val_set = ham_val_set + spam_val_set
    random.shuffle(train_set)
    random.shuffle(val_set)

    return train_set, val_set


def predict(x, w): return 1 / (1 + np.exp(-x@w))


def train_lr(file_list, num_iterations, learning_rate=0.01, regularization_param=2, *, debug=False, model='bow'):
    vectorizer = CountVectorizer(input='filename',
                                 decode_error='replace',
                                 analyzer=stemmed_words,
                                 stop_words=stopwords.words('english') + ['Subject'])
    x = vectorizer.fit_transform(file_list).toarray()
    if model == 'bow':
        pass
    elif model == 'bernoulli':
        binarize(x, threshold=0.0, copy=True)
    else:
        raise ValueError("The 'model' arg can only take values 'bow' and 'bernoulli'.")
    x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)

    num_features = x.shape[1]
    y = create_classes(file_list)

    # Initialize the weights
    w = np.random.normal(size=num_features)

    # Implementing the logistic regression algorithm
    for iter_ctr in range(num_iterations):
        y_pred = predict(x, w)
        error = y - y_pred
        if debug:
            print("Training error before iter {}: {}".format(iter_ctr+1,
                                                             np.sum(np.square(error))))

        # Weight update
        delta_grad = learning_rate*(x.transpose()@error - regularization_param*w)
        if debug:
            print("Grad change after iter {}: {}\n".format(iter_ctr+1, np.sum(delta_grad)))  # For debugging purposes
        w += delta_grad

    return vectorizer, w


def apply_lr(file_name_list, vectorizer, w):
    x = vectorizer.transform(file_name_list).toarray()
    x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
    y_pred = predict(x, w)
    binarize(x, threshold=0.5, copy=True)

    return y_pred


def test_classifier(test_file_list, vectorizer, w):
    y_test_true = create_classes(test_file_list)
    y_test_pred = apply_lr(test_file_list, vectorizer, w)
    error = np.sum(np.abs(y_test_pred - y_test_true))
    return 1 - error/len(y_test_true)


def learn_reg_const(train_file_path, reg_const_list=(0.1, 0.5, 1, 5, 10, 50, 100), num_iterations=45, *, model='bow'):
    # text_file_path = r'C:\Users\kxg220013\Documents\Machine Learning\Naive-Bayes-Classifier\data\train'
    file_list = [os.path.join(train_file_path, file_name) for file_name in os.listdir(train_file_path)
                 if file_name.endswith('.txt')]
    train_set, val_set = train_val_split(file_list)

    max_accuracy = 0
    best_reg_const = 0
    for reg_const in reg_const_list:
        vectorizer, w = train_lr(train_set,
                                 num_iterations=num_iterations,
                                 regularization_param=reg_const,
                                 debug=False, model=model)
        accuracy = test_classifier(val_set, vectorizer, w)
        if max_accuracy < accuracy:
            max_accuracy = accuracy
            best_reg_const = reg_const
    print(max_accuracy)
    return best_reg_const
