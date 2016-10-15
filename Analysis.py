
import os
# import pandas as pd
# import numpy as np
# import sklearn
import nltk
# from nltk.corpus import stopwords
# from nltk.stem.snowball import SnowballStemmer
import string
# from collections import Counter
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import LabelEncoder
# from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import classification_report as clsr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cross_validation
from sklearn.cross_validation import KFold, cross_val_score
# from NLTKPreprocessor import NLTKPreprocessor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# from sklearn import preprocessing
# import pylab as pl
from sklearn import svm
# from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import metrics
import pandas

class MyEvaluation(object):
    def __init__(self):
        self._accuracy = 0.0
        self._AUC = 0.0
        self._macroF1 = 0.0
        self._microF1 = 0.0
        self._F1 = []
        self._Precision = []
        self._Recall = []
        self._Conf_Matrix= pandas.DataFrame

    def _getAccuracy(self):
        return self._accuracy

    def _setAccuracy(self,value):
        self._accuracy=value

    def _getAUC(self):
        return self._AUC

    def _setAccuracy(self, value):
        self._AUC = value

    def _getAccuracy(self):
        return self._microF1

    def _setAccuracy(self, value):
        self._microF1 = value

    def _getAccuracy(self):
        return self._macroF1

    def _setAccuracy(self, value):
        self._macroF1 = value

    def _getPrecision(self):
        return self._Precision

    def _setPrecision(self, value):
        self._Precision = value

    def _getRecall(self):
        return self._Recall

    def _setRecall(self, value):
        self._Recall = value

    def _getConfusionMatrix(self):
        return self._Conf_Matrix

    def _setConfusionMatrix(self, value):
        self._Conf_Matrix = value




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
    # print(evalReport)
    return


def evaluate_nFoldCV(dataset,classlabels,classifier,n):
    X_train = dataset
    y_train = classlabels
    X_test = X_train
    y_test = y_train
    # k_fold = KFold(len(y_test), n_folds=n, shuffle=True, random_state=0)
    predictions=cross_validation.cross_val_predict(classifier,X_train,y_test,cv=n)
    # scores = cross_val_score(classifier, X_test, y_test, cv=n, n_jobs=1)
    accuracy= metrics.accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average=None)
    recall = recall_score(y_test, predictions, average=None)
    cm = confusion_matrix(y_test, predictions)
    myEval = MyEvaluation()
    myEval._setAccuracy(accuracy.mean())
    myEval._setPrecision(precision)
    myEval._setRecall(recall)
    return myEval



def evaluate_trainTestSplit(dataset,classLabels,classifier,testPercantage):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(dataset, classLabels, test_size=testPercantage,random_state=0)
    predictions = classifier.predict(X_test)  # Perform classification on an array of test vectors X.
    # evalReport = classification_report(y_test, predictions, target_names=classifier.classes_)
    precision=precision_score(y_test, predictions,average=None)
    recall=recall_score(y_test, predictions,average=None)
    # print(precision_recall_fscore_support(y_test, predictions, average='micro',labels =classLabels))
    cm = confusion_matrix(y_test, predictions)
    accuracy = classifier.score(X_test, y_test)
    myEval = MyEvaluation()
    myEval._setAccuracy(accuracy)
    myEval._setPrecision(precision)
    myEval._setRecall(recall)
    myEval._setConfusionMatrix(cm)
    return myEval


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
docTermMatrix = tfidf.fit_transform((open(f,encoding=None).read() for f in fileNames))
dataset=docTermMatrix

#print(docTermMatrix[1,:].todense())
#print(type(docTermMatrix))
#print(docTermMatrix.shape)
#print(classLabels)

classifier =MultinomialNB().fit(dataset, classLabels)

# test on training set---------------------------------------------------------

evaluate_testOnTrainingSet(dataset,classLabels,classifier)

# train-test split 60%-40% with multinominal_nb---------------------------------

percentage=0.4

myEval_trainTestSplit_MB=evaluate_trainTestSplit(dataset,classLabels,classifier,percentage)
print("\n split 60%-40% with Multinominal NB \n" )
print("Accuracy :" , myEval_trainTestSplit_MB._getAccuracy())
print("Precision:" , myEval_trainTestSplit_MB._getPrecision())
print("Recall   :" , myEval_trainTestSplit_MB._getRecall())
print("Confusion Matrix :\n " , myEval_trainTestSplit_MB._getConfusionMatrix())

# nFold CV with multinominal_nb-------------------------------------------------

n=10

myEval_nFoldCV_MB=evaluate_nFoldCV(dataset,classLabels,classifier,n)
print("\n 10 fold CV with Multinominal NB \n")
print("Accuracy :", myEval_nFoldCV_MB._getAccuracy())
print("Precision:", myEval_nFoldCV_MB._getPrecision())
print("Recall   :", myEval_nFoldCV_MB._getRecall())



# train-test split 60%-40% with SVM--------------------------------

clf = svm.SVC(kernel='linear',C=1.0,decision_function_shape=None)
classifier = clf.fit(dataset , classLabels)
myEval_trainTestSplit_SVM=evaluate_trainTestSplit(dataset,classLabels,classifier,percentage)
print("\n split 60%-40% with SVM \n" )
print("Accuracy :", myEval_trainTestSplit_SVM._getAccuracy())
print("Precision:", myEval_trainTestSplit_SVM._getPrecision())
print("Recall   :", myEval_trainTestSplit_SVM._getRecall())
print("Confusion Matrix :\n " , myEval_trainTestSplit_SVM._getConfusionMatrix())


