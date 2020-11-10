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

        #print(self.activationVectorsList)
   
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

    def setInitialWeights(self, initialWeights):
        self.weightsMatrixList = initialWeights

    def getSigmoid(self, z):
        return 1/(1 + np.exp(-z))
        
    def getError(self, pred_y, or_y):
        J = np.sum(-or_y*np.log(pred_y) - (1 - or_y)*np.log(1 - pred_y))/len(or_y)
        S=0
        for layer in range(self.layersNumber-1):
            S += np.sum(self.weightsMatrixList[layer][:, 1:]*self.weightsMatrixList[layer][:, 1:])
        S = S*(self.regularizationFactor)/(2*len(or_y))
        error = J+S
        return error

    def backPropagation(self, predY, oriY):
        print("RODANDO BACKPROPAGATION")
        for index in range(oriY.shape[0]):
            print("CALCULANDO DELTAS DA INSTANCIA", index, ":")
            self.calculateLastLayerDelta(predY[index], oriY[index])
            for layer in reversed(range(self.lastLayer())):             #decreasing
                self.deltaVectorsList[layer] = np.dot(self.weightsMatrixList[layer].T, self.deltaVectorsList[layer+1])
                self.deltaVectorsList[layer] = self.deltaVectorsList[layer]*self.activationVectorsList[layer]*(1-self.activationVectorsList[layer])
                self.deltaVectorsList[layer] = np.delete(self.deltaVectorsList[layer], 0, axis=0) #no delta for bias neurons
                print("delta", layer+1, ":\n", self.deltaVectorsList[layer])
                
            for layer in reversed(range(self.lastLayer())):             #decreasing
                self.gradientMatrixList[layer] = self.gradientMatrixList[layer] + np.dot(self.deltaVectorsList[layer+1], self.activationVectorsList[layer].T)
                print("GRADIENTE DE TETA", layer+1, "COM BASE NA INSTANCIA", index, ":\n", self.gradientMatrixList[layer])
        
        print("DATASET COMPLETO PROCESSADO. CALCULANDO GRADIENTES REGULARIZADOS")

        for layer in range(self.lastLayer()):             #decreasing
            self.gradientMatrixList[layer] = (1/len(oriY))*(self.gradientMatrixList[layer] + self.regularizationFactor*self.weightsMatrixList[layer])
            print("GRADIENTES FINAIS PARA TETA", layer+1, "(com regularizacão):\n", self.gradientMatrixList[layer])

        for layer in range(self.lastLayer()):             #decreasing
            self.weightsMatrixList[layer] = self.weightsMatrixList[layer] - self.learningRate*self.gradientMatrixList[layer]
    
    def train(self, trainDataFrame):
        oriY = trainDataFrame[trainDataFrame.columns[-1*(self.layersStructure[self.lastLayer()]):]]     
        oriY = np.array(oriY.to_numpy())
        trainDataFrame = trainDataFrame.iloc[:, :-1*(self.layersStructure[self.lastLayer()])]

        inputs = trainDataFrame.to_numpy()

        
        predY=[]
        
        for index in range(len(inputs)):
            print("------------------------------------------------------------")
            predicted = self.propagate(inputs[index])
            print("VALOR PREDITO: ", predicted)
            print("VALOR ESPERADO: ", oriY[index])
            print("ERRO J DA INSTANCIA", index, ":", self.getError(predicted, oriY[index]))

            predY.append(predicted)

        predY = np.array(predY)[:,:,0]

        print("J TOTAL DO DATASET (COM REGULARIZACÃO): ", self.getError(predY, oriY))
        print("------------------------------------------------------------")

        self.backPropagation(predY, oriY)

    
    def isLastButOneLayer(self, layer):
        return layer == (self.layersNumber-2)

    def propagate(self, input):
        print("PROPAGANDO ENTRADA: ", input)
        input = np.insert(input, 0, np.array([1]), axis=0)  #insert bias
        self.activationVectorsList[0] = np.expand_dims(input, axis=1)
        for layer in range(self.layersNumber-1):
            print("A", layer+1, ":\n", self.activationVectorsList[layer])
            z = np.dot(self.weightsMatrixList[layer], self.activationVectorsList[layer])
            print("Z", layer+2, ":\n", z)
            if(self.isLastButOneLayer(layer)):
                self.activationVectorsList[layer+1] = self.getSigmoid(z)
                print("F(X) =\n", self.activationVectorsList[layer+1])
                return self.activationVectorsList[layer+1]
            else:
                self.activationVectorsList[layer+1] = np.insert(self.getSigmoid(z), 0, 1, axis=0)

    def calculateLastLayerDelta(self, predY, oriY):
        self.deltaVectorsList[self.layersNumber-1] = np.array(predY - oriY,ndmin=2)
        print("delta", self.layersNumber, ":\n", self.deltaVectorsList[self.layersNumber-1])

    def lastLayer(self):
        return self.layersNumber-1