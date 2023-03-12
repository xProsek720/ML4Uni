# Autor projektu: Filip Wałęga
import numpy as np
import random as rn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Zadanie 1. Wczytaj do programu zbiór danych o kwiatach Iris.

df = pd.read_csv('pliki/iris/iris.csv')

# Zadanie 2. Wykonaj analizę danych zbioru Iris.

print(df.head())
df.plot(kind = "scatter", x="sepal.length", y="sepal.width")
#plt.show()

sns.pairplot(data=df, hue="variety")
#plt.show()

# Zadanie 3. Zaimplementuj algorytm tasowania, normalizacji, podziału zbioru na treningowy i walidacyjny.

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
            tmp+=(v1[1][i]-v2[i])**2                    # v[1][i] ponieważ dostaje tuple zamiast dataframe
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

DataProcessor.shuffle(df)
DataProcessor.normalization(df)
print(df.head())

train, test = DataProcessor.split(df, 0.7)
print(len(test), len(train))
print(KNN.clustering(test.iloc[0], train, 4))
print(f"Variety: {test.iloc[0].variety}")

print("\n"*10)
# Zadanie i+1. XD Zaimplementuj algorytm inferencji zbiorami miękkimi dla dla klasyfikacji wybranych produktów w sklepie.

traits = ("czerwone", "zielone", "okrągłe", "szpiczaste", "słodkie", "ostre")

products = {
    "onion" : (0, 1, 1, 0, 0, 0),
    "paprica" : (1, 0, 0, 0, 1, 1),
    "carrot" : (0, 0, 0, 1, 1, 0)
}

client = ("czerwone", "słodkie", "ostre")
client_w_input = (0.3, 0.3, 0.4)
client_traits = [0, 0, 0, 0, 0, 0]
client_weights = [0, 0, 0, 0, 0, 0]

for i in range(len(traits)):
    if traits[i] in client:
        index_trait = client.index(traits[i])
        client_weights[i] = client_w_input[index_trait]
        client_traits[i] = 1


#print(client_traits)
cpSum ={}
for product in products:
    cpSum[product] = 0
    for i in range(len(traits)):
        if products[product][i] == client_traits[i]:
            if products[product][i] == 1:
                cpSum[product] += 1*client_weights[i]

print(max(cpSum), cpSum)


