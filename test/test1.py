#%%
import os
from os import listdir
from os.path import isfile
import re

#%%
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
#%%
def collect_data_from(parent_dir, newsgroup_list):
    data = []
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()
    for group_id, newsgroup in enumerate(newsgroup_list):
        label = group_id
        dir_path = parent_dir + '/' + newsgroup + '/'
        files = [(filename, dir_path + filename)
                for filename in listdir(dir_path)
                if isfile(dir_path + filename)]
        files.sort()
        for filename, filepath in files:
            with open(filepath) as f:
                text = f.read().lower()
                # remove stop words
                words = [stemmer.stem(word) for word in re.split('\W+',text)]

#%%
filepath = '../datasets/20news-bydate/20news-bydate-train/comp.graphics/37913'
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
with open(filepath) as f:
    text = f.read().lower()
    # print(text)
    # print(re.split('\W+', text))
    words = [stemmer.stem(word) for word in re.split('\W+',text)]
    print(words)
# %%
