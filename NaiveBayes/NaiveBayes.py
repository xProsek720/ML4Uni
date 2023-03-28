import pandas as pd
import numpy as np

df = pd.read_csv("iris.csv")
#print(df.isnull())
irisTrain = df.sample(frac=0.8)
irisTest = df.drop(irisTrain.index)
#print(len(irisTest), len(irisTrain))
class NaiveBayes:
    @staticmethod
    def mean(atr):
        return sum(atr)/len(atr)
    @staticmethod
    def stdev(atr, mean):
        tmp = 0
        for i in atr:
            tmp +=(i-mean)**2
        return np.sqrt(tmp/len(atr))
    @staticmethod
    def propability(atr, mean, std):
        exponent = np.exp(-0.5*((atr - mean)/std)**2)
        return exponent/np.sqrt(2*np.pi*std**2)                             #wydaje mi się, że błąd jest w powyższych
    @staticmethod                                                           #metodach, ale nie jestem pewien
    def determineFlower(wynik):
        # ###
        # Metoda zwraca odmiane kwiatu iris
        # ###
        max = -1000
        max_name = None
        #print(wynik)
        for flo in wynik:
            if max < wynik[flo]:
                max = wynik[flo]
                max_name = flo
                #print(wynik[flo], flo)

        return [max, max_name]
        pass
    @staticmethod
    def classify(sample, irisTrain):
        #print(type(sample))
        wynik = {}
        for flower in irisTrain['variety'].unique():
             wynik[flower] = 0
        for flower in irisTrain['variety'].unique():
            #print(irisTrain.columns[0:4])
            for col in irisTrain.columns[0:4]:                              #0 do 4 bo inaczej był error od typu variety
                data = irisTrain[irisTrain['variety'] == flower]
                #print(data)
                atr = data.loc[:, col].values                               #values, aby nie brało indexów pod uwagę
                #print(atr)
                    #print("CDSACDAS","\n", "\n",atr)
                #print(data.iloc[-1])
                try:
                    #print(len(atr))
                    mean = NaiveBayes.mean(atr)
                    stv1 = NaiveBayes.stdev(atr, mean)
                    p1 = NaiveBayes.propability(sample[col], mean, stv1)
                    #print(col, atr[1])
                    wynik[flower] += p1
                    #print(wynik)
                except Exception:                                           #to było potrzebne przed użyciem iT.cols[0:4]
                    print(atr)
                    continue
        return NaiveBayes.determineFlower(wynik)

proc = {True: 0, False:0}
for i in range(len(irisTest.index)):
    index_X = i
    wyn = NaiveBayes.classify(irisTest.iloc[index_X], irisTrain)
    print(f"{wyn[1]:10} == {irisTest.iloc[index_X].variety:10} is {str(irisTest.iloc[index_X].variety == wyn[1]):5} \tWynik: [{wyn[0]:6f}]")
    #print(wyn[0], wyn[1], irisTest.iloc[index_X].variety)
    #print(irisTest.iloc[index_X].variety == wyn[1])
    proc[irisTest.iloc[index_X].variety == wyn[1]] +=1

print(proc)
print(f"Zgodność: {proc[True]/(proc[True]+proc[False])}%")
