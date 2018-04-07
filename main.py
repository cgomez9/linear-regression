import sys
import csv

from LinearRegression import LinearRegression

inputFile = sys.argv[1]
outputFile = sys.argv[2]
features = []

with open(inputFile, 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        features += row

regression = LinearRegression(outputFile)
regression.train(features)
