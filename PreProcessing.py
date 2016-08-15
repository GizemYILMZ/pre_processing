# -*- coding: utf-8 -*-
import nltk
from nltk.tokenize import word_tokenize
from stop_words import get_stop_words
import sklearn
import sklearn.feature_extraction
from TurkishStemmer import TurkishStemmer

stop_words = get_stop_words('tr')
stemmer = TurkishStemmer()
vectorizer = sklearn.feature_extraction.text.CountVectorizer(min_df=1)

def stem_word(word):
    stemmed_word= stemmer.stem(word)
    print(stemmed_word)

def remove_stop_words(word):
    if word.lower() not in stop_words:
        stem_word(word)
        return;

def tokenize_line(line):
        word_list=word_tokenize(line);
        return word_list;

i=0
with open('C:/1150haber/raw_texts/ekonomi/1.txt', encoding='utf-8') as file:
       for line in file:
            word_list=tokenize_line(line)
            for i in range(len(word_list)):
                remove_stop_words(word_list[i])


