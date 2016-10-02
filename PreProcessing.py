# -*- coding: utf-8 -*-

import string
import sklearn.feature_extraction
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from stop_words import get_stop_words
from TurkishStemmer import TurkishStemmer
from os import walk
import pandas as pd
from nltk.book import *
import snowballstemmer

vectorizer = sklearn.feature_extraction.text.CountVectorizer(min_df=1)
stemmed_list = " "
corpus = []
word_list = []


def remove_stop_words(stemmed_word):
    global stemmed_list
    global word_list
    stop_words = get_stop_words('tr')
    exclude = set(string.punctuation)
    if stemmed_word.lower() not in stop_words:
        if stemmed_word.lower() not in exclude:
            stemmed_list = stemmed_list + " " + stemmed_word
            word_list.append(stemmed_word.lower())
        return;
    return;


def stem_word(word):
  # stemmer = TurkishStemmer()
    stemmer2 = snowballstemmer.stemmer('turkish');
    stemmed_word = stemmer2.stemWord(word)
    remove_stop_words(stemmed_word)
    return;


def tokenize_line(line):
    word_list = word_tokenize(line);
    return word_list;


def create_corpus(file_path):
    global corpus
    with open(file_path, encoding='utf-8') as open_file:
        for line in open_file:
            word_list = tokenize_line(line)
            for i in range(len(word_list)):
                stem_word(word_list[i])
    open_file.close()
    corpus.append(stemmed_list)
    new_path=file_path + "kopya"
    print(new_path)
    target = open(new_path, 'w', encoding='utf-8')
    target.write(stemmed_list)
    target.close()
    return


# find file names in a directory
f = []
for (dirpath, dirnames, filenames) in walk('C:/1150haber/raw_texts/test'):
    f.extend(filenames)
    break

# create corpus for all files in a directory
for i in range(len(filenames)):
    stemmed_list = ''
    word_list.clear()
    file_path = dirpath + "/" + filenames[i]
    print(file_path)
    create_corpus(file_path)
    fdist1 = FreqDist(word_list)
    print(fdist1.most_common(10))

target = open('target.txt', 'w')

# create document - term matrix
X = vectorizer.fit_transform(corpus).toarray()

print('X: {0}'.format(X))
print('vectorizer.vocabulary_: {0}'.format(vectorizer.vocabulary_))
target.write('X: {0}'.format(X))
target.write(format(vectorizer.vocabulary_))

pd.options.display.max_rows = 150
pd.options.display.max_columns= 7483
countvec = CountVectorizer()
A = pd.DataFrame(countvec.fit_transform(corpus).toarray(), columns=countvec.get_feature_names())
print(A)


