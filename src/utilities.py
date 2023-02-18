# Utility functions for the odd jobs
import os
import shutil
from zipfile import ZipFile
from fnmatch import fnmatch


def unzip_folder(zip_path, extract_path):
    print(zip_path)
    with ZipFile(zip_path, 'r') as zObject:
        zObject.extractall(path=extract_path)


def extract_text_files(zip_files, extract_path):
    """
    Extract text files to a temporary directory and then, move them to
    the final location for processing. Delete the temporary location after the move.
    """
    temp_dir = extract_path + r'\temp'

    if not os.path.isdir(extract_path):
        os.mkdir(extract_path)
        print("Created extract path: {}".format(extract_path))

    for zip_file in zip_files:
        if not os.path.isdir(temp_dir):
            os.mkdir(temp_dir)
            print("Created temporary directory {} for zip_file {}".format(temp_dir, zip_file))

        unzip_folder(zip_file, temp_dir)
        print("Successfully unzipped folder")

        pattern = "*.txt"
        flist = [os.path.join(path_, name) for path_, _, files in os.walk(temp_dir)
                 for name in files if fnmatch(name, pattern)]
        count = 0
        for source_file in flist:
            dest_file = extract_path + '\\' + os.path.basename(source_file)
            if os.path.isfile(dest_file):
                print("File {} already exists. Skipping...".format(dest_file))
                continue
            os.replace(source_file, dest_file)
            count += 1
        print("Successfully moved {} files".format(count))

        shutil.rmtree(temp_dir)
        print("Successfully deleted temporary directory {}".format(temp_dir))


if __name__ == '__main__':
    # create all files in one place.
    text = []
    path = r'C:\Users\kxg220013\Documents\Machine Learning\Naive-Bayes-Classifer\data\train'
    zip_files = ['enron1_train.zip', 'enron2_train.zip', 'enron4_train.zip']
    zip_files = [(path+'\\'+zip_file) for zip_file in zip_files]
    extract_text_files(zip_files, path)