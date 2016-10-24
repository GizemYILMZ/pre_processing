
import os
import pandas as pd
import numpy as np
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
from sklearn.metrics import f1_score

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

    def _setAUC(self, value):
        self._AUC = value

    def _getPrecision(self):
        return self._Precision

    def _setPrecision(self, value):
        self._Precision = value

    def _getRecall(self):
        return self._Recall

    def _setRecall(self, value):
        self._Recall = value

    def _getF1(self):
        return self._F1

    def _setF1(self, value):
        self._F1 = value

    def _getF1_macro(self):
        return self._F1_macro

    def _setF1_macro(self, value):
        self._F1_macro = value

    def _getF1_micro(self):
        return self._F1_micro

    def _setF1_micro(self, value):
        self._F1_micro = value

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
    # print(stems)
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
    # k_fold = KFold(len(y_test), n_folds=n, shuffle=True, random_state=0)
    predictions=cross_validation.cross_val_predict(classifier,dataset,classlabels,cv=n)
    # score=cross_val_score(classifier, dataset, classlabels,cv=n,scoring='accuracy')  avg is same as metrics.accuracy_score

    accuracy = metrics.accuracy_score(classlabels, predictions)
    precision = precision_score(classlabels, predictions, average=None)
    recall = recall_score(classlabels, predictions, average=None)

    F1=metrics.f1_score(classlabels, predictions, average=None)
    F1_macro=metrics.f1_score(classlabels, predictions, average='macro')
    F1_micro=metrics.f1_score(classlabels, predictions, average='micro')

    cm = confusion_matrix(classlabels, predictions)

    myEval = MyEvaluation()
    myEval._setAccuracy(accuracy.mean())
    myEval._setPrecision(precision)
    myEval._setRecall(recall)
    myEval._setF1(F1)
    myEval._setF1_macro(F1_macro)
    myEval._setF1_micro(F1_micro)
    myEval._setConfusionMatrix(cm)

    return myEval



def evaluate_trainTestSplit(dataset,classLabels,classifier,testPercantage):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(dataset, classLabels, test_size=testPercantage,random_state=0)

    predictions = classifier.predict(X_test)

    accuracy = classifier.score(X_test, y_test)
    precision = precision_score(y_test, predictions, average=None)
    recall = recall_score(y_test, predictions, average=None)

    F1=metrics.f1_score(y_test, predictions, average=None)
    F1_macro=metrics.f1_score(y_test, predictions, average='macro')
    F1_micro=metrics.f1_score(y_test, predictions, average='micro')

    cm = confusion_matrix(y_test, predictions)

    myEval = MyEvaluation()
    myEval._setAccuracy(accuracy)
    myEval._setPrecision(precision)
    myEval._setRecall(recall)
    myEval._setF1(F1)
    myEval._setF1_macro(F1_macro)
    myEval._setF1_micro(F1_micro)
    myEval._setConfusionMatrix(cm)

    return myEval


path = 'data/news/'
corpus = []

classLabels = []
fileNames = []
for subdir, dirs, files in os.walk(path):
    for file in files:
        file_path = subdir + os.path.sep + file
        fileNames.append(file_path)
        classLabels.append(subdir[15:])

print("i m here 1 ")
tfidf = TfidfVectorizer(tokenizer=tokenize, preprocessor=preprocess, lowercase=True, stop_words='english')
docTermMatrix = tfidf.fit_transform((open(f,encoding='utf8').read() for f in fileNames))
dataset=docTermMatrix

# print(docTermMatrix[1,:].todense())
# print(type(docTermMatrix))
#print(docTermMatrix.shape)
#print(classLabels)
print("i m here 2 ")
classifier =MultinomialNB().fit(dataset, classLabels)

# test on training set---------------------------------------------------------
print("i m here 3 ")
evaluate_testOnTrainingSet(dataset,classLabels,classifier)

# train-test split 60%-40% with multinominal_nb---------------------------------

percentage=0.4
print("i m here 4 ")
myEval_trainTestSplit_MNB=evaluate_trainTestSplit(dataset,classLabels,classifier,percentage)
print("\n ---------------------split 60%-40% with Multinominal NB ------------------\n" )
print("Accuracy :" , myEval_trainTestSplit_MNB._getAccuracy())
print("Precision:" , myEval_trainTestSplit_MNB._getPrecision())
print("Recall   :" , myEval_trainTestSplit_MNB._getRecall())
print("F1       :" , myEval_trainTestSplit_MNB._getF1())
print("F1 macro :" , myEval_trainTestSplit_MNB._getF1_macro())
print("F1 micro :" , myEval_trainTestSplit_MNB._getF1_micro())
print("Confusion Matrix :\n " , myEval_trainTestSplit_MNB._getConfusionMatrix())

# nFold CV with multinominal_nb-------------------------------------------------

n=10

myEval_nFoldCV_MNB=evaluate_nFoldCV(dataset,classLabels,classifier,n)
print("\n -----------------" , n ,"fold CV with Multinominal NB----------------------- \n")
print("Accuracy :", myEval_nFoldCV_MNB._getAccuracy())
print("Precision:", myEval_nFoldCV_MNB._getPrecision())
print("Recall   :", myEval_nFoldCV_MNB._getRecall())
print("F1       :" , myEval_nFoldCV_MNB._getF1())
print("F1 macro :" , myEval_nFoldCV_MNB._getF1_macro())
print("F1 micro :" , myEval_nFoldCV_MNB._getF1_micro())
print("Confusion Matrix :\n " , myEval_nFoldCV_MNB._getConfusionMatrix())


# train-test split 60%-40% with SVM--------------------------------

clf = svm.SVC(kernel='linear',C=1.0,decision_function_shape=None)
classifier = clf.fit(dataset , classLabels)

myEval_trainTestSplit_SVM=evaluate_trainTestSplit(dataset,classLabels,classifier,percentage)
print("\n -------------------split 60%-40% with SVM------------------------ \n" )
print("Accuracy :", myEval_trainTestSplit_SVM._getAccuracy())
print("Precision:", myEval_trainTestSplit_SVM._getPrecision())
print("Recall   :", myEval_trainTestSplit_SVM._getRecall())
print("F1       :" , myEval_trainTestSplit_SVM._getF1())
print("F1 macro :" , myEval_trainTestSplit_SVM._getF1_macro())
print("F1 micro :" , myEval_trainTestSplit_SVM._getF1_micro())
print("Confusion Matrix :\n " , myEval_trainTestSplit_SVM._getConfusionMatrix())


myEval_nFoldCV_SVM=evaluate_nFoldCV(dataset,classLabels,classifier,n)
print("\n -----------------" , n ,"fold CV with SVM------------------------ \n")
print("Accuracy :", myEval_nFoldCV_SVM._getAccuracy())
print("Precision:", myEval_nFoldCV_SVM._getPrecision())
print("Recall   :", myEval_nFoldCV_SVM._getRecall())
print("F1       :" , myEval_nFoldCV_SVM._getF1())
print("F1 macro :" , myEval_nFoldCV_SVM._getF1_macro())
print("F1 micro :" , myEval_nFoldCV_SVM._getF1_micro())
print("Confusion Matrix :\n " , myEval_nFoldCV_SVM._getConfusionMatrix())