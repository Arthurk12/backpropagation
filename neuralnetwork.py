import numpy as np
import pandas as pd

class NeuralNetwork:

    def __init__(self, layersStructure, regularizationFactor, learningRate):
        self.layersStructure = layersStructure
        self.layersNumber = len(layersStructure)
        self.regularizationFactor = regularizationFactor
        self.learningRate = learningRate

        self.weightsMatrixList = []
        self.activationVectorsList = []
        self.deltaVectorsList = []
        self.gradientMatrixList = []

        
        self.initRandomWeightsMatrixes()
        self.initActivationVectorsList()
        self.initDeltaVectorsList()
        self.initGradientMatrixes()
   

    def initRandomWeightsMatrixes(self):
        for index in range(self.layersNumber-1):
            weightsMatrix = np.random.rand(self.layersStructure[index+1], self.layersStructure[index]+1)
            self.weightsMatrixList.append(weightsMatrix)

    def initActivationVectorsList(self):
        for index in range(self.layersNumber):
            if(index == (self.layersNumber-1)):
                activationVector = np.zeros((self.layersStructure[index], 1))
            else:
                activationVector = np.zeros((self.layersStructure[index]+1, 1))
            self.activationVectorsList.append(activationVector)
    
    def initDeltaVectorsList(self):
        for index in range(self.layersNumber):
            deltaVector = np.zeros((self.layersStructure[index], 1))
            self.deltaVectorsList.append(deltaVector)

    def initGradientMatrixes(self):
        for index in range(self.layersNumber-1):
            gradient = np.zeros((self.layersStructure[index+1], 1), self.layersStructure[index]+1)
            self.gradientMatrixList.append(gradient)

    def getSigmoid(self, z):
        return 1/(1 + np.exp(-z))
        
    def getError(self, or_y, pred_y, weightsMatrixList):
        J = np.sum(-or_y*np.log(pred_y) - (1 - or_y)*np.log(1 - pred_y))/len(or_y)

        for layer in range(self.layersNumber-1):
            S += np.sum(self.weightsMatrixList[layer][:, 1:]*self.weightsMatrixList[layer][:, 1:])
        S = S*(self.regularizationFactor)/(2*len(or_y))
        error = J+S
        return error

    def backPropagation(self, predicted_originalY):
        #print("----- BACKPROPAGATION -----")
        for predY, oriY in predicted_originalY:
            
            print("DELTAS")
            self.calculateLastLayerDelta(oriY, predY)
            for layer in reversed(range(self.layersNumber-1)):             #decreasing
                #print("LAYER: ", layer)
                #print("1: ", self.weightsMatrixList[layer].T, self.deltaVectorsList[layer+1])
                self.deltaVectorsList[layer] = np.dot(self.weightsMatrixList[layer].T, self.deltaVectorsList[layer+1])
                #print("2: ",self.deltaVectorsList[layer])
                self.deltaVectorsList[layer] = self.deltaVectorsList[layer]*self.activationVectorsList[layer]*(1-self.activationVectorsList[layer])
                #print("3: ",self.deltaVectorsList[layer])
                self.deltaVectorsList[layer] = np.delete(self.deltaVectorsList[layer], 0, axis=0) #no delta for bias neurons
                #print("4: ",self.deltaVectorsList[layer])
            
            print("GRADIENTS")
            for layer in reversed(range(self.layersNumber-1)):             #decreasing
                print("LAYER: ", layer)
                print("5: ", np.dot(self.deltaVectorsList[layer+1], self.activationVectorsList[layer].T))
                self.gradientMatrixList[layer] = self.gradientMatrixList[layer] + np.dot(self.deltaVectorsList[layer+1], self.activationVectorsList[layer].T)
        
        print("FINAL GRADIENTS")
        for layer in reversed(range(self.layersNumber-1)):             #decreasing
            self.gradientMatrixList[layer] = (1/len(predicted_originalY))*(self.gradientMatrixList[layer] + self.regularizationFactor*self.weightsMatrixList[layer])

        print("UPDATE WEIGHTS")
        for layer in reversed(range(self.layersNumber-1)):             #decreasing
            self.weightsMatrixList[layer] = self.weightsMatrixList[layer] - self.learningRate*self.gradientMatrixList[layer]
    
    def train(self, trainDataFrame):
        oriY = trainDataFrame.drop('target', 1)
        del trainDataFrame['target']
        
        inputs = trainDataFrame.to_numpy()
        biasVector = np.ones((inputs.shape[0], 1))
        inputs = np.hstack((biasVector, inputs))
        
        print(inputs.T)
        self.activationVectorsList[0] = inputs.T
        print(self.activationVectorsList)
        self.propagate()
        
        #self.backPropagation(1,2)

    
    def isLastButOneLayer(self, layer):
        return layer != (self.layersNumber-2)

    def propagate(self):
        for layer in range(self.layersNumber-1):
            z = np.dot(self.weightsMatrixList[layer], self.activationVectorsList[layer])

            if(self.isLastButOneLayer(layer)):
                self.activationVectorsList[layer+1] = np.insert(self.getSigmoid(z), 0, 1, axis=0)
            else:
                self.activationVectorsList[layer+1] = self.getSigmoid(z)

    def calculateLastLayerDelta(self, oriY, predY):
        self.deltaVectorsList[self.layersNumber-1] = np.array([[predY - oriY]])