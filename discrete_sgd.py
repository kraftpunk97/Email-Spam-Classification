from src.sgdclassifier import *
from src.utilities import *
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


def get_file_list(text_file_path): return [os.path.join(text_file_path, file_name) for file_name in os.listdir(text_file_path)
                                           if file_name.endswith('.txt')]


def main(train_path, test_path, zip_file_name):
    zip_file_name_train = train_path + zip_file_name + '_train.zip'
    zip_file_name_test = test_path + zip_file_name + '_test.zip'

    extract_text_files([zip_file_name_train], train_path)
    extract_text_files([zip_file_name_test], test_path)
    vectorizer, clf = get_classifier(train_path, model='bernoulli')

    x_train_file_list = get_file_list(train_path)
    x_test_file_list = get_file_list(test_path)
    y_test_true = create_classes(x_test_file_list)
    x_test = vectorizer.transform(x_test_file_list)
    y_test_pred = clf.predict(x_test)

    accuracy = accuracy_score(y_true=y_test_true, y_pred=y_test_pred)
    f1 = f1_score(y_true=y_test_true, y_pred=y_test_pred)
    recall = recall_score(y_true=y_test_true, y_pred=y_test_pred)
    precision = precision_score(y_true=y_test_true, y_pred=y_test_pred)

    print("Stats for Discrete SGDClassifier...")
    print("Accuracy: {}".format(accuracy))
    print("F1: {}".format(f1))
    print("Recall: {}".format(recall))
    print("Precision: {}".format(precision))

    for file in x_test_file_list:
        os.remove(file)
    for file in x_train_file_list:
        os.remove(file)


if __name__ == '__main__':
    train_path = 'data/train/'
    test_path = 'data/test/'
    zip_file_names = ["enron1", "enron2", "enron4"]
    for zip_file in zip_file_names:
        main(train_path, test_path, zip_file)