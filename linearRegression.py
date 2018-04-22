import numpy as np
import matplotlib.pyplot as plt
import time
import csv

class LinearRegression:

    def __init__(self,learningRate,writer):
        self.betasVector = np.array([0,0,0],float)
        self.exampleVector = []
        self.trueLabels = []
        self.learningRate = learningRate
        self.spamwriter = writer

    def train(self,iterations,examples):
        if len(examples) > 0:
            self.scale(examples)
            for example in examples:
                self.trueLabels.append(example[2])
                example = np.delete(example, 2)
                scaledExample = np.concatenate((np.array([1]),example))
                self.exampleVector.append(scaledExample)
            for i in range(iterations):
                difference = self.updateBetas()
                if abs(difference) < 0.0001:
                    break
            b0 = str(self.betasVector[0])
            b1 = str(self.betasVector[1])
            b2 = str(self.betasVector[2])
            self.spamwriter.writerow((str(self.learningRate),str(iterations),b0,b1,b2))
        else:
            print("No features provided.")

    def scale(self,examples):
        examples[:,:2] = (examples[:,:2] - np.mean(examples[:,:2], axis =0)) / np.std(examples[:,:2], axis=0)

    def updateBetas(self):
        n = len(self.exampleVector)
        previousBetas = np.copy(self.betasVector)
        for betaIndex,beta in enumerate(self.betasVector):
            sumatory = 0
            for i in range(n):
                sumatory += (self.f(i,previousBetas)-self.trueLabels[i])*self.exampleVector[i][betaIndex]
            self.betasVector[betaIndex] -= self.learningRate * (1/n) * sumatory
        return previousBetas[0] - self.betasVector[0]

    def f(self,xi,previousBetas):
        b1 = previousBetas[1]*self.exampleVector[xi][1]
        b2 = previousBetas[2]*self.exampleVector[xi][2]
        return previousBetas[0] + b1 + b2
