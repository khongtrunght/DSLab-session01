from os.path import isfile
from os import listdir
import os

def gather_20newsgroups_data():
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, '../datasets/20news-bydate/')
    # path = '../datasets/20news-bydate/'
    dirs = [path + sub_dir_name + '/' 
            for sub_dir_name in listdir(path)
            if not isfile(path + sub_dir_name) ]

    train_dir, test_dir = (dirs[0], dirs[1]) if 'train' in dirs[0] \
        else (dirs[1], dirs[0])
    # get list of the newsgorups in train_dir
    list_newsgroups = [newsgroup
                        for newsgroup in  listdir(train_dir)]
    list_newsgroups.sort()
    return list_newsgroups

if __name__ == '__main__':
    news = gather_20newsgroups_data()
    print(news)

