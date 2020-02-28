import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.base import clone

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier

from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, roc_auc_score, balanced_accuracy_score

import load_dataset
from MPCL import MPCL
from rDEP import DEP, EnsembleTransform
from HoTdiagram import HoTdiagram

def EvalClassifiers(Name, Classifiers, X, y, n_splits=10, score = balanced_accuracy_score):
    df = pd.DataFrame()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
    for train_index, test_index in skf.split(X, y):
        df_sim = pd.DataFrame()
        Xtr, Xte = X[train_index], X[test_index]
        ytr, yte = y[train_index], y[test_index]
        # Process the data
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)
        for name, clf in Classifiers:
            try:
                clone_clf = clone(clf)
                clone_clf.fit(Xtr,ytr)
                y_pred = clone_clf.predict(Xte)
                df_sim[name] = [score(yte,y_pred)]
            except:
                print("Classifier %s failed to process dataset %s" % (name,Name))
        df = pd.concat([df,df_sim])
    df.to_csv("CSVs/%s.csv" % Name)
    return df

VotingSVC = VotingClassifier([("RBF SVC",SVC(gamma="scale")),
             ("Linear SVC",SVC(kernel="linear")),
             ("Poly SVC",SVC(kernel="poly"))])
BaggingSVC = BaggingClassifier(base_estimator=SVC(gamma="scale"),n_estimators=10, random_state=0)
Classifiers = [("Linear SVC",SVC(kernel="linear",gamma="scale")), 
               ("RBF SVC",SVC(gamma="scale")),
               ("Poly SVC",SVC(kernel="poly",gamma="scale")),
               ("SVC Ensemble",VotingSVC),
               ("Bagging SVC",BaggingSVC),
               ("MPCL",MPCL(Nsamples=0)), 
               ("DEP",DEP()),
               ("r-DEP (Ensemble)",make_pipeline(EnsembleTransform(VotingSVC),StandardScaler(),DEP())),
               ("r-DEP (Bagging)",make_pipeline(EnsembleTransform(BaggingSVC),StandardScaler(),DEP())),
              ]

def GetSize(dataset):
    return datasets.fetch_openml(dataset[1],version=dataset[2],return_X_y = True)[0].shape[0]

AllDataSets = [
    ("Breast Cancer Wisconsin","wdbc",1),
    ("Diabetes","diabetes",1),
    ("Banknote","banknote-authentication",1),
    ("Spambase","spambase",1),
    ("Ionosphere","ionosphere",1),
    ("Colic","colic",2),
    ("Sonar","sonar",1),
    ("Tic-Tac-Toe","tic-tac-toe",1),
    ("Monks-2","monks-problems-2",1),
    ("Australian","Australian",4),
    ("Banana","banana",1),
    ("Cylinder Bands","cylinder-bands",2),
    ("Chess","kr-vs-kp",1),
    ("Haberman","haberman",1),
    ("Mushroom","mushroom",1),
    ("Phoneme","phoneme",1),
    ("Titanic","Titanic",2),
    ("Pishing Websites","PhishingWebsites",1),
    ("Internet Advertisements","Internet-Advertisements",2),
    ("Thoracic Surgery","thoracic_surgery",1),
    ("Credit Approval","credit-approval",1),
    ("Hill-Valley","hill-valley",1),
    ("Egg-Eye-State","eeg-eye-state",1),
    ("MOFN-3-7-10","mofn-3-7-10",1),
    ("Credit-g","credit-g",1),
    ("Accute Inflammations","acute-inflammations",1),
    ("ilpd","ilpd",1),
    ("Arsene","arcene",1),
    ("Blood Transfusion","blood-transfusion-service-center",1),
    ("Steel Plates Fault","steel-plates-fault",1),
    ("Sick","sick",1)
#     Large Datasets
#     ("Adult","Adult",2),
#     ("Cover Type","covertype",2),
#     ("Bank Marketing","bank-marketing",1),
#     ("Adult Census","adult-census",1),
#     ("Gisette","gisette",2),
    ]

# Sort datasets from number of samples
AllDataSets.sort(key=GetSize)


data = pd.DataFrame()
for name, dataset, version in AllDataSets:
    start_time = time.time()
    print("\nProcessing dataset: ",name)
    X, y = datasets.fetch_openml(dataset,version=version,return_X_y = True)
    # Imput missing data
    X = SimpleImputer().fit_transform(X)
    df = EvalClassifiers(name,Classifiers, X, y).rename(index={0:name})
    # HoTdiagram(df, PlotName = name, significance_level = 0.95, Gaussian = True, NormalizeData=False)
    data = pd.concat([data,df])
    data.to_csv("CSVs/BinaryDataSets.csv")
    print("\nTime to process the dataset: %2.2f seconds." % (time.time() - start_time))

# plt.figure(figsize=(10, 8))
# HoTdiagram(df, PlotName = "BinaryDatasets", significance_level = 0.95, Gaussian = False, NormalizeData=False)
# df.mean()
