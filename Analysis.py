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

class MyEval:
    accuracy=0.0
    AUC=0.0
    F1=0.0
    def __init__(self):
        return
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


def evaluate_testOnTrainingSet(dataset,classlabels,classifier):
    X_train = dataset
    y_train = classlabels
    X_test = X_train
    y_test = y_train
    predictions = classifier.predict(X_test)
    evalReport = classification_report(y_test, predictions, target_names=classifier.classes_)
    print(evalReport)
    return


def evaluate_nFoldCV(dataset,classlabels,classifier,n):
    X_train = dataset
    y_train = classlabels
    X_test = X_train
    y_test = y_train
    # k_fold = KFold(len(y_test), n_folds=n, shuffle=True, random_state=0)
    scores = cross_val_score(classifier, X_test, y_test, cv=n, n_jobs=1)
    print(scores)
    print("Mean score: {0:.3f} (+/-{1:.3f})".format(scores.mean(), scores.std()))
    return


def evaluate_trainTestSplit(dataset,classLabels,classifier,testPercantage):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(dataset, classLabels, test_size=testPercantage,random_state=0)
    predictions = classifier.predict(X_test)  # Perform classification on an array of test vectors X.
    evalReport = classification_report(y_test, predictions, target_names=classifier.classes_)
    cm = confusion_matrix(y_test, predictions)
    print(evalReport)
    print("Confusion matrix:")
    accuracy = classifier.score(X_test, y_test)
    print(accuracy)
    return


path = 'data/1150haber/'
corpus = []

classLabels = []
fileNames = []
for subdir, dirs, files in os.walk(path):
    for file in files:
        file_path = subdir + os.path.sep + file
        fileNames.append(file_path)
        classLabels.append(subdir[15:])


tfidf = TfidfVectorizer(tokenizer=tokenize, preprocessor=preprocess, lowercase=True, stop_words='english')
docTermMatrix = tfidf.fit_transform((open(f,encoding='utf8').read() for f in fileNames))

#print(docTermMatrix[1,:].todense())
#print(type(docTermMatrix))
#print(docTermMatrix.shape)
#print(classLabels)
dataset=docTermMatrix

classifier =MultinomialNB().fit(dataset, classLabels)

# test on training set
evaluate_testOnTrainingSet(dataset,classLabels,classifier)

# train-test split 60%-40% with multinominal_nb
percentage=0.4
print(" split 60%-40% with Multinominal NB")
evaluate_trainTestSplit(dataset,classLabels,classifier,percentage)

# nFold CV with multinominal_nb
n=10
print(" 10 fold CV with Multinominal NB")
evaluate_nFoldCV(dataset,classLabels,classifier,n)



# clf = svm.SVC(kernel='linear',C=1.0,decision_function_shape=None)
# clf.fit(X_train, y_train)
# prediction=clf.predict(X_test)
# accuracy=clf.score(X_test,prediction)
# print(accuracy)