import sys
import csv
import numpy as np

from linearRegression import LinearRegression

inputFile = 'input2.csv'#sys.argv[1]
outputFile = 'output.csv'#sys.argv[2]
examples = []

#with open(inputFile, 'r') as csvfile:
    #spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    #for row in spamreader:
        #examples += row

regression = LinearRegression(outputFile)
examples = np.loadtxt ( inputFile, delimiter= ',' , skiprows=1)
regression.train(examples)
