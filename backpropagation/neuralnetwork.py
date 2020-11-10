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

        self.float_formatter = "{:.5f}".format
        np.set_printoptions(formatter={'float_kind':self.float_formatter})
        pd.options.display.float_format = self.float_formatter

        #print(self.activationVectorsList)
   
    def initRandomWeightsMatrixes(self):
        for index in range(self.layersNumber-1):
            weightsMatrix = np.random.rand(self.layersStructure[index+1], self.layersStructure[index]+1)
            self.weightsMatrixList.append(weightsMatrix)
    
    def initActivationVectorsList(self):
        for index in range(self.layersNumber):
            self.activationVectorsList.append(0)

    def initDeltaVectorsList(self):
        for index in range(self.layersNumber):
            self.deltaVectorsList.append(0)

    def initGradientMatrixes(self):
        for index in range(self.layersNumber-1):
            self.gradientMatrixList.append(0)

    def setInitialWeights(self, initialWeights):
        self.weightsMatrixList = initialWeights

    def getSigmoid(self, z):
        return 1/(1 + np.exp(-z))
        
    def getError(self, pred_y, or_y):
        J = np.sum(-or_y*np.log(pred_y) - (1.0 - or_y)*np.log(1.0 - pred_y))
        return J
    
    def getErrorWithRegularization(self, pred_y, or_y):
        J = self.getError(pred_y, or_y)/len(or_y)
        S=0
        for layer in range(self.layersNumber-1):
            S += np.sum(self.weightsMatrixList[layer][:,1:]*self.weightsMatrixList[layer][:,1:])    #weight matrix without bias weights
        S = S*(self.regularizationFactor)/(2*len(or_y))
        error = J+S
        return error

    def backPropagation(self, predY, oriY):
        print("RODANDO BACKPROPAGATION")
        
        for index in range(len(oriY)):
            print("\n\nCALCULANDO DELTAS DA INSTANCIA", index, ":")
            self.calculateLastLayerDelta(predY[index], oriY[index])
            for layer in reversed(range(self.lastLayer())):             #decreasing
                self.deltaVectorsList[layer] = np.dot(self.weightsMatrixList[layer].T, self.deltaVectorsList[layer+1])
                self.deltaVectorsList[layer] = np.multiply(self.deltaVectorsList[layer],np.multiply(self.activationVectorsList[layer],np.subtract(1.0,self.activationVectorsList[layer])))
                self.deltaVectorsList[layer] = np.delete(self.deltaVectorsList[layer], 0, axis=0) #no delta for bias neurons
                print("delta", layer+1, ":\n", self.deltaVectorsList[layer])
                
            for layer in reversed(range(self.lastLayer())):             #decreasing
                print("GRADIENTE DE TETA", layer+1, "COM BASE NA INSTANCIA", index, ":\n", np.dot(self.deltaVectorsList[layer+1], self.activationVectorsList[layer].T))
                self.gradientMatrixList[layer] = np.add(self.gradientMatrixList[layer], np.dot(self.deltaVectorsList[layer+1], self.activationVectorsList[layer].T))
        
        print("\n\nDATASET COMPLETO PROCESSADO. CALCULANDO GRADIENTES REGULARIZADOS\n")

        for layer in range(self.lastLayer()):             #decreasing
            P = self.regularizationFactor*self.weightsMatrixList[layer]
            P[:,0]=0
            self.gradientMatrixList[layer] = (1/len(oriY))*np.add(self.gradientMatrixList[layer], P)
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
            print("VALOR PREDITO: \n", predicted)
            print("VALOR ESPERADO: \n", oriY[index])
            print("ERRO J DA INSTANCIA", index, ":", self.getError(predicted, oriY[index]))

            predY.append(predicted)

        predY = np.array(predY)

        print("J TOTAL DO DATASET (COM REGULARIZACÃO): ", self.getErrorWithRegularization(predY, oriY))
        print("------------------------------------------------------------")


        self.backPropagation(predY, oriY)
    
    def isLastButOneLayer(self, layer):
        return layer == (self.layersNumber-2)

    def propagate(self, input):
        print("PROPAGANDO ENTRADA: ", input)
        input = np.array(input, ndmin=2)
        input = np.insert(input, 0, np.array([1]), axis=1)  #insert bias
        self.activationVectorsList[0] = input.T             #transforma em vetor coluna

        for layer in range(self.layersNumber-1):
            print("A", layer+1, ":\n", self.activationVectorsList[layer])
            z = np.dot(self.weightsMatrixList[layer], self.activationVectorsList[layer])
            print("Z", layer+2, ":\n", z)
            if(self.isLastButOneLayer(layer)):
                self.activationVectorsList[layer+1] = self.getSigmoid(z)
                print("F(X) =\n", self.activationVectorsList[layer+1])
                return np.squeeze(self.activationVectorsList[layer+1])
            else:
                self.activationVectorsList[layer+1] = np.insert(self.getSigmoid(z), 0, 1, axis=0)

    def calculateLastLayerDelta(self, predY, oriY):
        self.deltaVectorsList[self.lastLayer()] = np.array(predY - oriY,ndmin=2).T
        print("delta", self.layersNumber, ":\n", self.deltaVectorsList[self.layersNumber-1])

    def lastLayer(self):
        return self.layersNumber-1