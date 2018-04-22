import sys
import csv
import numpy as np

from linearRegression import LinearRegression

inputFile = 'input2.csv'
outputFile = 'output.csv'
examples = []

learningRates = [0.5]

with open(outputFile, 'w') as csvfile:
    spamwriter = csv.writer(csvfile, lineterminator='\n', delimiter=',')
    for learningRate in learningRates:
        print("Parar learning rate: ",learningRate)
        print("")
        regression = LinearRegression(learningRate=learningRate, writer=spamwriter)
        examples = np.loadtxt ( inputFile, delimiter= ',' , skiprows=1)
        regression.train(100,examples)
        print("")
    #regression = LinearRegression(learningRate=0.5, writer=spamwriter)
    #examples = np.loadtxt ( inputFile, delimiter= ',' , skiprows=1)
    #regression.train(10,examples)
