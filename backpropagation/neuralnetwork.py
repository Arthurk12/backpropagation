import numpy as np
import pandas as pd

class NeuralNetwork:

    def __init__(self, layersStructure, regularizationFactor, learningRate):
        self.layersStructure = layersStructure
        self.layersNumber = len(layersStructure)
        self.regularizationFactor = regularizationFactor
        self.learningRate = learningRate
        self.epsilon = 0.000001

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
        J = np.sum((-or_y)*np.log(pred_y) - (1.0 - or_y)*np.log(1.0 - pred_y))

        J = J/len(or_y)
        return J

    def getErrorWithRegularization(self, pred_y, or_y):
        J = self.getError(pred_y, or_y)

        S=0
        for layer in range(self.lastLayer()):
            S += np.sum(self.weightsMatrixList[layer][:,1:]*self.weightsMatrixList[layer][:,1:])    #weight matrix without bias weights
        S = S*((self.regularizationFactor)/(2*len(or_y)))

        error = J+S
        return error
    
    def backPropagation(self, inputs, oriY):
        print("RODANDO BACKPROPAGATION")
        
        for index in range(len(inputs)):
            predicted = self.propagate(inputs[index], False)
            self.calculateLastLayerDelta(predicted, oriY[index])
            for layer in reversed(range(self.lastLayer())):             #decreasing
                self.deltaVectorsList[layer] = np.dot(self.weightsMatrixList[layer].T, self.deltaVectorsList[layer+1])
                self.deltaVectorsList[layer] = np.multiply(self.deltaVectorsList[layer],np.multiply(self.activationVectorsList[layer],np.subtract(1.0,self.activationVectorsList[layer])))
                self.deltaVectorsList[layer] = np.delete(self.deltaVectorsList[layer], 0, axis=0) #no delta for bias neurons
                print("delta", layer+1, ":\n", self.deltaVectorsList[layer])
                
            for layer in reversed(range(self.lastLayer())):             #decreasing
                print("GRADIENTE DE THETA", layer+1, "COM BASE NA INSTANCIA", index, ":\n", np.dot(self.deltaVectorsList[layer+1], self.activationVectorsList[layer].T))
                self.gradientMatrixList[layer] = np.add(self.gradientMatrixList[layer], np.dot(self.deltaVectorsList[layer+1], self.activationVectorsList[layer].T))
        
        print("\n\nDATASET COMPLETO PROCESSADO. CALCULANDO GRADIENTES REGULARIZADOS\n")

        for layer in range(self.lastLayer()):             #decreasing
            P = self.regularizationFactor*self.weightsMatrixList[layer]
            P[:,0]=0
            self.gradientMatrixList[layer] = (1/len(oriY))*np.add(self.gradientMatrixList[layer], P)
            print("GRADIENTES FINAIS PARA THETA", layer+1, "(com regularizacão):\n", self.gradientMatrixList[layer])

        print(self.calculateNumericGradients(inputs, oriY))

        for layer in range(self.lastLayer()):             #decreasing
            self.weightsMatrixList[layer] = self.weightsMatrixList[layer] - self.learningRate*self.gradientMatrixList[layer]
    
    def train(self, trainDataFrame):
        oriY = trainDataFrame[trainDataFrame.columns[-1*(self.layersStructure[self.lastLayer()]):]]     
        oriY = np.array(oriY.to_numpy())
        trainDataFrame = trainDataFrame.iloc[:, :-1*(self.layersStructure[self.lastLayer()])]

        inputs = trainDataFrame.to_numpy()

        predY=[]
        
        print("------------------------------------------------------------")
        print("CALCULANDO ERRO")
        for index in range(len(inputs)):
            predicted = self.propagate(inputs[index], True)
            print("VALOR PREDITO: \n", predicted)
            print("VALOR ESPERADO: \n", oriY[index])
            print("ERRO J DA INSTANCIA", index, ":", self.getError(predicted, np.atleast_2d(oriY[index])))

            predY.append(predicted)

        predY = np.array(predY)
        print("PREDY VECTOR: ", predY)
        print("ORIY VECTOR: ", oriY)

        print("J TOTAL DO DATASET (COM REGULARIZACÃO): ", self.getErrorWithRegularization(predY, oriY))
        print("------------------------------------------------------------")


        self.backPropagation(inputs, oriY)
    
    def isLastButOneLayer(self, layer):
        return layer == (self.layersNumber-2)

    def propagate(self, input, verbose):
        if(verbose):
            print("PROPAGANDO ENTRADA: ", input)
        input = np.array(input, ndmin=2)
        input = np.insert(input, 0, np.array([1]), axis=1)  #insert bias
        self.activationVectorsList[0] = input.T             #transforma em vetor coluna

        for layer in range(self.layersNumber-1):
            if(verbose):
                print("A", layer+1, ":\n", self.activationVectorsList[layer])
            z = np.dot(self.weightsMatrixList[layer], self.activationVectorsList[layer])
            if(verbose):
                print("Z", layer+2, ":\n", z)
            if(self.isLastButOneLayer(layer)):
                self.activationVectorsList[layer+1] = self.getSigmoid(z)
                if(verbose):
                    print("F(X) =\n", self.activationVectorsList[layer+1])
                return np.atleast_1d(np.squeeze(self.activationVectorsList[layer+1]))
            else:
                self.activationVectorsList[layer+1] = np.insert(self.getSigmoid(z), 0, 1, axis=0)

    def calculateLastLayerDelta(self, predY, oriY):
        self.deltaVectorsList[self.lastLayer()] = np.array(predY - oriY,ndmin=2).T
        print("delta", self.layersNumber, ":\n", self.deltaVectorsList[self.layersNumber-1])

    def lastLayer(self):
        return self.layersNumber-1
    def calculateNumericGradients(self, inputs, oriY):
        gradients = []
        for layer in range(self.lastLayer()):
            gradients.append(np.zeros(self.weightsMatrixList[layer].shape))
        
        originalWeightsMatrixList = self.weightsMatrixList

        for layer in reversed(range(self.lastLayer())):
            for row in range(self.weightsMatrixList[layer].shape[0]):
                for column in range(self.weightsMatrixList[layer].shape[1]):
                    self.weightsMatrixList[layer][row,column] += self.epsilon    
                    plus = []
                    for index in range(len(inputs)):
                        plus.append(self.propagate(inputs[index], False))
                    
                    self.weightsMatrixList[layer][row,column] -= 2*self.epsilon       
                    minus = []
                    for index in range(len(inputs)):
                        minus.append(self.propagate(inputs[index], False))
                    
                    plus = np.array(plus)
                    minus = np.array(minus)

                    #print("CONJ 1: ", plus)
                    #print("CONJ 2: ", minus)
                    #print("plus: ", self.getErrorWithRegularization(plus, oriY))
                    #print("minus: ", self.getErrorWithRegularization(minus, oriY))
                    gradients[layer][row,column] = (self.getErrorWithRegularization(plus, oriY)-self.getErrorWithRegularization(minus, oriY))/(2*self.epsilon)
                    self.weightsMatrixList = originalWeightsMatrixList


        return gradients
