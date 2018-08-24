import numpy as np
import matplotlib.pyplot as plt
import time
import csv

class LinearRegression:

    def __init__(self,learning_rate,writer):
        self.betas_vector = np.array([0,0,0],float)
        self.example_vector = []
        self.true_labels = []
        self.learning_rate = learning_rate
        self.spamwriter = writer

    def train(self,iterations,examples):
        if len(examples) > 0:
            self.scale(examples)
            for example in examples:
                self.true_labels.append(example[2])
                example = np.delete(example, 2)
                scaled_example = np.concatenate((np.array([1]),example))
                self.example_vector.append(scaled_example)
            for i in range(iterations):
                difference = self.update_betas()
                if abs(difference) < 0.0001:
                    break
            b0 = str(self.betas_vector[0])
            b1 = str(self.betas_vector[1])
            b2 = str(self.betas_vector[2])
            self.spamwriter.writerow((str(self.learning_rate),str(iterations),b0,b1,b2))
        else:
            print("No features provided.")

    def scale(self,examples):
        examples[:,:2] = (examples[:,:2] - np.mean(examples[:,:2], axis =0)) / np.std(examples[:,:2], axis=0)

    def update_betas(self):
        n = len(self.example_vector)
        previous_betas = np.copy(self.betas_vector)
        for beta_index,beta in enumerate(self.betas_vector):
            sumatory = 0
            for i in range(n):
                sumatory += (self.f(i,previous_betas)-self.true_labels[i])*self.example_vector[i][beta_index]
            self.betas_vector[beta_index] -= self.learning_rate * (1/n) * sumatory
        return previous_betas[0] - self.betas_vector[0]

    def f(self,xi,previous_betas):
        b1 = previous_betas[1]*self.example_vector[xi][1]
        b2 = previous_betas[2]*self.example_vector[xi][2]
        return previous_betas[0] + b1 + b2
