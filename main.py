import sys
import csv
import numpy as np

from linearRegression import LinearRegression

input_file = 'input2.csv'
output_file = 'output.csv'
examples = []

learning_rates = [0.5]

with open(output_file, 'w') as csvfile:
    spamwriter = csv.writer(csvfile, lineterminator='\n', delimiter=',')
    for learning_rate in learning_rates:
        regression = LinearRegression(learning_rate=learning_rate, writer=spamwriter)
        examples = np.loadtxt ( input_file, delimiter= ',' , skiprows=1)
        regression.train(100,examples)
    #regression = LinearRegression(learning_rate=0.5, writer=spamwriter)
    #examples = np.loadtxt ( input_file, delimiter= ',' , skiprows=1)
    #regression.train(10,examples)
