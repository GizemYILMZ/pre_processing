# -*- coding: utf-8 -*-
import nltk
import sklearn
import string
import sklearn.feature_extraction
from nltk.tokenize import word_tokenize
from stop_words import get_stop_words
from TurkishStemmer import TurkishStemmer
from os import walk

vectorizer = sklearn.feature_extraction.text.CountVectorizer(min_df=1)

stemmed_list = " "
corpus = []
def stem_word(word):
    global stemmed_list
    stemmer = TurkishStemmer()
    stemmed_word = stemmer.stem(word)
    stemmed_list = stemmed_list + " " + stemmed_word
    return stemmed_list

def remove_stop_words(word):
    stop_words = get_stop_words('tr')
    exclude = set(string.punctuation)
    if word.lower() not in stop_words and word not in exclude:
        stem_word(word)
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
                remove_stop_words(word_list[i])
    open_file.close()
    corpus.append(stemmed_list)
    return


# find file names in a directory
f = []
for (dirpath, dirnames, filenames) in walk('C:/1150haber/raw_texts/ekonomi'):
    f.extend(filenames)
    break

# create corpus for all files in a directory
for i in range(len(filenames)):
    stemmed_list = ''
    file_path=dirpath + "/" + filenames[i]
    print(file_path)
    create_corpus(file_path)

target = open('target.txt', 'w')

# create document - term matrix
X = vectorizer.fit_transform(corpus).toarray()

print('X: {0}' .format(X))
print('vectorizer.vocabulary_: {0}'.format(vectorizer.vocabulary_))
target.write('X: {0}'.format(X))
target.write(format(vectorizer.vocabulary_))



