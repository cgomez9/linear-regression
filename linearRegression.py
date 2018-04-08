class LinearRegression:

    def __init__(self,outputFile):
        self.weigthVectors = np.array([0,0,0])
        self.exampleVector = []
        self.trueLabels = []
        self.outputFile = outputFile

    def train(self,features):
        with open(self.outputFile, 'w') as csvfile:
            spamwriter = csv.writer(csvfile, lineterminator='\n')
            if len(examples) > 0:
                for example in examples:
                    features = example.split(',')
                    self.exampleVector.append((1,int(features[0]),int(features[1])))
                    self.trueLabels.append(int(features[2]))
                while(True):
                    pass
            else:
                print("No features provided.")
