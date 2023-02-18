import os
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer

stemmer = PorterStemmer()
analyzer = CountVectorizer().build_analyzer()
def stemmed_words(string): return (stemmer.stem(word) for word in analyzer(string))


def create_classes(file_name_list): return np.array([('spam' in file_name) for file_name in file_name_list],
                                                    dtype=np.int64)


def get_classifier(train_file_path, model='bow'):
    #train_file_path = r'C:\Users\kxg220013\Documents\Machine Learning\Naive-Bayes-Classifier\data\train'
    file_list = [os.path.join(train_file_path, file_name) for file_name in os.listdir(train_file_path)
                 if file_name.endswith('.txt')]

    if model == 'bow':
        binary = False
    elif model == 'bernoulli':
        binary = True
    else:
        raise ValueError("The 'model' arg can only take values 'bow' and 'bernoulli'.")

    vectorizer = CountVectorizer(input='filename', decode_error='replace',
                                 analyzer=stemmed_words, binary=binary,
                                 stop_words=stopwords.words('english') + ['Subject'])

    # Using Log Loss (For Logistic Regression) and  L2 Regularization
    # as per project description
    sgd_clf = SGDClassifier(loss="log_loss", penalty="l2",
                            alpha=0.01, max_iter=45,
                            learning_rate='constant')

    x = vectorizer.fit_transform(file_list)
    y = create_classes(file_list)

    parameters = {'eta0': (0.1, 0.5, 1, 5, 10)}
    clf = GridSearchCV(sgd_clf, parameters)
    clf.fit(x, y)

    return vectorizer, clf
