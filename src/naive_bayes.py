import os
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import binarize
from sklearn.feature_extraction.text import CountVectorizer

stemmer = PorterStemmer()
analyzer = CountVectorizer().build_analyzer()
def stemmed_words(string): return (stemmer.stem(word) for word in analyzer(string))


def create_classes(text_file_path):
    file_name_list = [file_name for file_name in os.listdir(text_file_path) if file_name.endswith('.txt')]
    res_vec = np.array([int('spam' in file_name) for file_name in file_name_list], dtype=np.int64)
    return res_vec


def train_nb(text_file_path, *, model='bow'):
    # text_file_path = r'C:\Users\kxg220013\Documents\Machine Learning\Naive-Bayes-Classifier\data\train'
    file_list = [os.path.join(text_file_path, file_name) for file_name in os.listdir(text_file_path)
                 if file_name.endswith('.txt')]

    if model == 'bernoulli':
        binary = True
    elif model == 'bow':
        binary = False
    else:
        raise ValueError("The 'model' arg can only take values 'bow' and 'bernoulli'.")

    vectorizer = CountVectorizer(input='filename',
                                 decode_error='replace',
                                 analyzer=stemmed_words, binary=binary,
                                 stop_words=stopwords.words('english')+['Subject'])
    x = vectorizer.fit_transform(file_list)

    y = create_classes(text_file_path)

    # Divide our corpus into ham and spam
    x_ham = x[y == 0]
    x_spam = x[y == 1]

    num_ham_mail = x_ham.shape[0]
    num_spam_mail = x_spam.shape[0]

    # Calculate the number of tokens of each word for each class
    ham_total_tokens = np.sum(x_ham, axis=0)
    spam_total_tokens = np.sum(x_spam, axis=0)

    # Calculate the conditional probability with Laplace smoothing
    cond_prob_ham = (ham_total_tokens+1) / (ham_total_tokens+1).sum()
    cond_prob_spam = (spam_total_tokens+1) / (spam_total_tokens+1).sum()

    # Calculate the class prior probability
    ham_prior = num_ham_mail/(num_ham_mail+num_spam_mail)
    spam_prior = num_spam_mail/(num_ham_mail+num_spam_mail)

    # Apply log to avoid underflow
    # Apply ravel to convert numpy.matrix to numpy.array
    cond_prob_spam = np.log(cond_prob_spam).ravel()
    cond_prob_ham = np.log(cond_prob_ham).ravel()
    ham_prior = np.log(ham_prior)
    spam_prior = np.log(spam_prior)

    return vectorizer, (cond_prob_ham, cond_prob_spam), (ham_prior, spam_prior)


def apply_nb(vectorizer, priors, cond_prob, text_file_list):
    x = vectorizer.transform(text_file_list).toarray()
    ham_prior, spam_prior = priors
    cond_prob_ham, cond_prob_spam = cond_prob

    ham_score = np.multiply(x, cond_prob_ham).sum(axis=1) + ham_prior     # For ham classification
    spam_score = np.multiply(x, cond_prob_spam).sum(axis=1) + spam_prior  # For spam classification
    y = (ham_score < spam_score).A1  # Obtaining our prediction and converting numpy.matrix to numpy.array

    return y
