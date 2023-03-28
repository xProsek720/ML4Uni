# Autor projektu: Filip Wałęga
import numpy as np
import random as rn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class DataProcessor:
    @staticmethod
    def shuffle(x):
        for i in range(len(x)):
            j:int = rn.randint(i, len(x)-1)
            x.iloc[i], x.iloc[j] = x.iloc[j], x.iloc[i]

    @staticmethod
    def normalization(x):
        #v' = (v-min)/(max-mix)
        values = x.select_dtypes(exclude="object")
        columnNames = values.columns.tolist()
        for column in columnNames:
            data = x.loc[:, column]
            maxn = max(data)
            minn = min(data)
            for row in range(0, len(x), 1):
                vprim=(x.at[row, column]-minn)/(maxn-minn)
                x.at[row, column] = vprim

    @staticmethod
    def split(x, k):
        n = int(k*len(x))
        return x[:n], x[n:]


class KNN:
    @staticmethod
    def distance(v1, v2):
        tmp = 0
        for i in range(len(v1[1])-1):
            tmp+=(v1[1][i]-v2[i])**2
        return tmp**(1/2)

    @staticmethod
    def clustering(sample, x, k):
        classes = {'Setosa': 0, 'Versicolor': 0, 'Virginica': 0}
        #wyznaczenie odleglosci miedzy probka a kazdym elementem bazy
        distances = []
        for row in x.iterrows():
            distances.append(KNN.distance(row, sample))
        x = x.assign(Distance=distances)

        #sortowanie zboiru x wzgledem odległości
        x.sort_values("Distance", inplace=True)


        #głosowanie w zależkości od k
        for i in range(k):
            classes[x.iloc[i].variety]+=1

        maxK = ('Setosa', 0)
        for key in classes.items():
            if key[1] > maxK[1]:
                maxK = key
        return maxK, classes
def toLOOP():

    df = pd.read_csv('pliki/iris/iris.csv')
    DataProcessor.shuffle(df)
    DataProcessor.normalization(df)
    #print(df.head())

    train, test = DataProcessor.split(df, 0.7)
    ret = KNN.clustering(test.iloc[0], train, 4)
    if test.iloc[0].variety == ret[0][0]:
        print(True)
        return True
    else:
        print(False)
        return False


zonk = 0
ilTestow = 1000

for i in range(ilTestow):
    ret = toLOOP()
    if ret:
        zonk += 1
print(zonk/ilTestow)

#print("\n"*10)
# Zadanie i+1. XD Zaimplementuj algorytm inferencji zbiorami miękkimi dla dla klasyfikacji wybranych produktów w sklepie.

df = pd.read_csv('pliki/iris/iris.csv')
DataProcessor.shuffle(df)
DataProcessor.normalization(df)
print(df.head())

train, test = DataProcessor.split(df, 0.7)
print(len(test), len(train))
print(KNN.clustering(test.iloc[0], train, 4))
print(f"Variety: {test.iloc[0].variety}")
