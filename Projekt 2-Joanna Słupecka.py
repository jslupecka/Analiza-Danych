
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib
import datetime
import os
import csv

pd.set_option('display.max_rows', None)
#wczytywanie danych
data = pd.read_csv('data.csv')
#zastepowanie etykiet M i B na 1 i 0
cleanup_nums = {"diagnosis":     {"M": 1, "B": 0}}
data = data.replace(cleanup_nums)
#liczenie korelacji
corrMatrix=data.corr()
s=corrMatrix.unstack()
so = s.sort_values(kind="quicksort", ascending=False)
print(so)
print(type(so))

data.drop(data.columns[0], axis=1, inplace=True)
data.drop(data.columns[len(data.columns)-1], axis=1, inplace=True)
X = data.loc[:, data.columns != "diagnosis"]
y = data.diagnosis
print(X.shape)
print(y.shape)
print(data.columns)

#dzielenie zbioru na treningowy i testowy 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#trenowanie sieci neuronowej
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(activation='relu', solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)
clf = clf.fit(X_train, y_train)


#liczenie zgodnosci modelu
y_pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score
print("Accuracy", accuracy_score(y_test, y_pred))
print("Error", (1-accuracy_score(y_test, y_pred)))

plt.show()