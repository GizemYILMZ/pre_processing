import os
import pandas as pd
import numpy as np
import sklearn
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import string
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report as clsr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cross_validation
from sklearn.cross_validation import KFold, cross_val_score
from NLTKPreprocessor import NLTKPreprocessor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import preprocessing
#import pylab as pl
from sklearn import svm

# ------------Preprocessing Data-------------------

def stemmerTrFps6(term):
    return term[:6]

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer(item))
    return stemmed

def tokenize(text):
    stemmer = stemmerTrFps6
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

def preprocess(term):
    return term.lower().translate( string.punctuation )

# returns the distinct elements from a list
def distinct(myList):
    seen = set()
    seen_add = seen.add
    return [x for x in myList if not (x in seen or seen_add(x))]

#-------------End Of Preprocessing Data------------------


# -------------- Classifiers-----------------------------

def multinomial_nb(X_train, y_train):
    classifier =MultinomialNB().fit(X_train, y_train)
    return classifier

def multinomial_nb_classifier(X_train, y_train,X_test,y_test):
    classifier=multinomial_nb(X_train,y_train)
    predictions = classifier.predict(X_test)  # Perform classification on an array of test vectors X.
    evalReport = classification_report(y_test, predictions, target_names=classifier.classes_)  # distinct(classLabels))
    print("Eval Report : ")
    print(evalReport)
    cm = confusion_matrix(y_test,predictions)  # Compute confusion matrix to evaluate the accuracy of a classification
    print("Confusion matrix:")
    print(cm)
    print("Accuracy :")
    accuracy = classifier.score(X_test, y_test)
    print(accuracy)
    return

def kfold_cross_validation(X_train, y_train,X_test,y_test):
    print("--------10 fold croos validation with Multinominal NB------------")
    k_fold = KFold(len(y_test), n_folds=10, shuffle=True, random_state=0)
    classifier = multinomial_nb(X_train,y_train)
    scores = cross_val_score(classifier, X_test, y_test, cv=k_fold, n_jobs=1)
    print(scores)
    print("Mean score: {0:.3f} (+/-{1:.3f})".format(scores.mean(), scores.std()))
    return

# -------------- End of Classifiers-----------------------------


path = 'data/1150haber/'
corpus = []

classLabels = []
fileNames = []
for subdir, dirs, files in os.walk(path):
    for file in files:
        file_path = subdir + os.path.sep + file
        fileNames.append(file_path)
        classLabels.append(subdir[15:])


# TfidfVectorizer : Convert a collection of raw documents to a matrix of TF-IDF features.
tfidf = TfidfVectorizer(tokenizer=tokenize, preprocessor=preprocess, lowercase=True, stop_words='english')

# fit_transform :  Learn vocabulary and idf, return term-document matrix.
# docTermMatrix : Tf-idf-weighted document-term matrix. (X)
docTermMatrix = tfidf.fit_transform((open(f,encoding='utf8').read() for f in fileNames))

#print(docTermMatrix[1,:].todense())
#print(type(docTermMatrix))
#print(docTermMatrix.shape)
#print(classLabels)

# test on training set
X_train = docTermMatrix
y_train = classLabels
X_test = docTermMatrix
y_test = classLabels

# multinominal_nb
multinomial_nb_classifier(X_train,y_train,X_test,y_test)

# train-test split 60%-40% with multinominal_nb
print(" split 60%-40% with Multinominal NB")
X_train, X_test, y_train, y_test = cross_validation.train_test_split(docTermMatrix, classLabels, test_size=0.4, random_state=0)
multinomial_nb_classifier(X_train,y_train,X_test,y_test)

# k_fold(10) cross validation  with multinominal_nb
X_train = docTermMatrix
y_train = classLabels
X_test = docTermMatrix
y_test = classLabels

kfold_cross_validation(X_train,y_train,X_test,y_test)


clf = svm.SVC(kernel='linear',C=1.0,decision_function_shape=None)
clf.fit(X_train, y_train)
prediction=clf.predict(X_test)
accuracy=clf.score(X_test,prediction)
print(accuracy)