import numpy as np
import matplotlib.pyplot as plt
import time
import csv

class LinearRegression:

    def __init__(self,outputFile):
        self.betasVector = np.array([0,0,0],float)
        self.exampleVector = []
        self.trueLabels = []
        self.outputFile = outputFile
        self.learningRate = 0.01

    def train(self,examples):
        with open(self.outputFile, 'w') as csvfile:
            spamwriter = csv.writer(csvfile, lineterminator='\n')
            if len(examples) > 0:
                for example in examples:
                    self.trueLabels.append(example[2])
                    example = np.delete(example, 2)
                    scaledExample = self.scale(example)
                    scaledExample = np.concatenate((np.array([1]),scaledExample))
                    self.exampleVector.append(scaledExample)
                previousBetas = np.copy(self.betasVector)
                self.plotExamples()
                for i in range(100):
                    self.updateBetas()
                    print(i,self.betasVector)
                    difference = previousBetas - self.betasVector
                    if abs(difference[0]) < 0.001:
                        break
            else:
                print("No features provided.")

    def scale(self,x):
        return (x - np.mean(x,axis=0)) / np.std(x,axis=0)

    def updateBetas(self):
        n = len(self.exampleVector)
        #print("HACIENDO UPDATE DE BETAS")
        sumatory = 0
        for i in range(n):
            #print("TrueLabel: ", self.trueLabels[i])
            #print("Example: ", self.exampleVector[i])
            sumatory += (self.f(i)-self.trueLabels[i])*self.exampleVector[i]
        #print("Learning rate", self.learningRate)
        #print("N", n)
        #print("Cuenta: ", self.learningRate * (1/n) * sumatory)
        np.subtract(self.betasVector, self.learningRate * (1/n) * np.array(sumatory),
                    out=self.betasVector,
                    casting="unsafe")
        #print("Nuevos betas: ", self.betasVector)

    def f(self,xi):
        #print("ENTRANDO EN F()",xi)
        b1 = self.betasVector[1]*self.exampleVector[xi]
        #print("B1: ", b1)
        b2 = self.betasVector[2]*self.exampleVector[xi]
        #print("B2: ", b2)
        #print("RESULT: ", self.betasVector[0] + b1 + b2)
        return self.betasVector[0] + b1 + b2

    def plotExamples(self):
        for index,example in enumerate(self.exampleVector):
            plt.plot(example[1], example[2], color='blue', marker='o')
        plt.show()
