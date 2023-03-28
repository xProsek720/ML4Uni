# Autor projektu: Filip Wałęga
import numpy as np
import random as rn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

###
# K-NN:
# ###

print("K-NN: ")

# Zadanie 1. Wczytaj do programu zbiór danych o kwiatach Iris.

df = pd.read_csv('pliki/iris/iris.csv')

# Zadanie 2. Wykonaj analizę danych zbioru Iris.

# Drukuje pierwsze pare wierszy, można podać konkretną ilość
print(df.head())

# Wbudowana metoda do robienia wykresu z danych DataFrame z modułu pandas
df.plot(kind = "scatter", x="sepal.length", y="sepal.width")
plt.show()     # To moje do wyświetlania wykresu w pycharm

# Piękna metoda z modułu seaborn do Analizy Danych, daje nam zbiór wykresów z całego DataFrame
sns.pairplot(data=df, hue="variety")
plt.show()

# Zadanie 3. Zaimplementuj algorytm tasowania, normalizacji, podziału zbioru na treningowy i walidacyjny.

class DataProcessor:
    ###
    # Klasa służąca do wstępnej 'obróbki' danych
    # ###

    @staticmethod
    def shuffle(x):
        ###
        # Metoda, która tasuje zbiór danych
        # ###
        for i in range(len(x)):
            j:int = rn.randint(i, len(x)-1)
            x.iloc[i], x.iloc[j] = x.iloc[j], x.iloc[i]

    @staticmethod
    def normalization(x):
        ###
        # Normalizacja danych do wartości z przedziału od 0 do 1
        # ###
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
        ###
        # Podział zbioru na części np. 0.7 => 0.7 train 0.3 test
        # ###
        n = int(k*len(x))
        return x[:n], x[n:]


class KNN:
    ###
    # K-Nearest Neighbour:
    # Algorytm stworzony głownie pod baze danych z irysami
    # W skrócie liczy dystans do wszystkich znanych wyników
    # Później bierze k minimum i z tego wyniku maksimum daje
    # nam decyzje jaki to kwiatek.
    # ###

    @staticmethod
    def distance(v1, v2):
        ###
        # Metoda licząca dystans między dwoma wektorami
        # z sqrt((xn+yn)**2 + ...) jak widać w kodzie
        # ###
        tmp = 0
        for i in range(len(v1[1])-1):
            tmp+=(v1[1][i]-v2[i])**2                    # v[1][i] ponieważ dostaje tuple zamiast dataframe, prawdopodobnie
        return tmp**(1/2)                               # przez x.iterrows(), wcześniej v1 iterowalo id i dystans wydawał się zwiekszać co wynik

    @staticmethod
    def clustering(sample, x, k):
        ###
        # Metoda odpowiedzialna za klasteryzacje danych
        # Mówiąc bardziej po polsku dzieli nam wyniki z DF na grupy
        # Od razu również analizuje próbkę i głosuję do której grupy
        # należy.
        # ###

        classes = {'Setosa': 0, 'Versicolor': 0, 'Virginica': 0}

        #wyznaczenie odleglosci miedzy probka a kazdym elementem bazy
        distances = []
        for row in x.iterrows():
            distances.append(KNN.distance(row, sample))
        x = x.assign(Distance=distances)

        #sortowanie zboiru x wzgledem odległości
        x.sort_values("Distance", inplace=True)

        #głosowanie w zależkości od k wraz z uzupełnieniem dict classes
        for i in range(k):
            classes[x.iloc[i].variety]+=1

        # Wybranie maxa, można to też zrobić max(dict key=dict.keys()) bodajże
        maxK = ('Setosa', 0)
        for key in classes.items():
            if key[1] > maxK[1]:
                maxK = key
        return maxK, classes


DataProcessor.shuffle(df)               # Tasacja bd Irysów
DataProcessor.normalization(df)         # Normalizacja bd Irysów
print(df.head())                        # Sprawdzenie danych

train, test = DataProcessor.split(df, 0.7)      # Podział zbioru na train i test
print(len(test), len(train))                    # Sprawdzenie czy poprawnie dzieli
print(KNN.clustering(test.iloc[0], train, 4))   # K-NN pierwszego sampla z części test max(wyników_z_min_odległości)
print(f"Variety: {test.iloc[0].variety}")       # Wydrukowanie prawdziwej odmiany sampla

print("\n"*10+"Zbiory Miękkie:")

###
# ZBIORY MIĘKKIE:
# ###

#       tzn 4. (BrainTiredException)
# Zadanie i+1. Zaimplementuj algorytm inferencji zbiorami miękkimi dla dla klasyfikacji wybranych produktów w sklepie.

# Tuple cech produktów wersja 'Human Readable'
traits = ("czerwone", "zielone", "okrągłe", "szpiczaste", "słodkie", "ostre")

# Dict produktów, w wersji, która jest wygodniejsza dla mnie i dla komputera
products = {
    "onion" : (0, 1, 1, 0, 0, 0),
    "paprica" : (1, 0, 0, 0, 1, 1),
    "carrot" : (0, 0, 0, 1, 1, 0)
}
# Są to dwa tuple reprezentujące wejście użytkownika
client = ("czerwone", "słodkie", "ostre")
client_w_input = (0.3, 0.3, 0.4)

# Inicjacja zmiennych 'przetłumaczonych' z 'clientInput' na 'shopInput'
client_traits = [0, 0, 0, 0, 0, 0]
client_weights = [0, 0, 0, 0, 0, 0]

# Translacja zachcianek klienta na te sklepowe
for i in range(len(traits)):
    if traits[i] in client:
        index_trait = client.index(traits[i])
        client_weights[i] = client_w_input[index_trait]
        client_traits[i] = 1


print(client_traits) #DEBUG :D

# Liczenie wszystkich produktów wg preferencji klienta
cpSum ={}
for product in products:
    cpSum[product] = 0
    for i in range(len(traits)):
        if products[product][i] == client_traits[i]:
            if products[product][i] == 1:
                cpSum[product] += 1*client_weights[i]


#Wydrukowanie najbardziej korzystnego produktu wg preferencji klienta
print(max(cpSum), cpSum)


