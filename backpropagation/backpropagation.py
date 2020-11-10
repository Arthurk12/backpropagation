import argparse
import pandas as pd
import numpy as np
from neuralnetwork import NeuralNetwork

def main(arglist):
    np.set_printoptions(formatter={'float_kind':"{:.5f}".format})

    regularizationFactor, layerNeuroniums = processNetworkFile(arglist.network)
    initialWeights = processInitialWeightsFile(arglist.initial_weights)
    dataset = pd.read_csv(arglist.dataset, sep=' |; |, ', header = None , engine='python')
       
    print("DATASET: \n", dataset)
    for i in range(len(initialWeights)):
        print("MATRIZ DE PESOS", i, ":\n", initialWeights[i])
    print("FATOR DE REGULARIZACÃO: \n", regularizationFactor)


    n1 = NeuralNetwork(layerNeuroniums, regularizationFactor, 0.25)
    n1.setInitialWeights(initialWeights)
    n1.train(dataset)



def processNetworkFile(networkFile):
    network = open(networkFile, "r")
    regularizationFactor = float(network.readline())
    
    nums = network.readlines()
    nums = [int(i) for i in nums]
    
    return regularizationFactor, np.array(nums)

def processInitialWeightsFile(initialWeightsFile):
    initialWeights = open(initialWeightsFile, "r")
    matrixes = []
    for line in initialWeights.readlines():
        line = line.split(";")
        arrays = []
        for row in line:
            arrays.append(np.fromstring(row, sep=','))
        matrixes.append(np.array(arrays))
    weightMatrixList = matrixes
    return weightMatrixList

def parse_arguments():

    parser = argparse.ArgumentParser(prog = 'Programa para validacão do algoritmo de backpropagation',
                                    description='Desenvolvido por Arthur Böckmann e João Dick\n',
                                    formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--network',
                        help = 'Arquivo que descreve a estrutura da rede',
                        required=True
    )
    parser.add_argument('--initial_weights',
                        help = 'Arquivo contendo os pesos iniciais utilizados pela rede',
                        required=True
    )
    parser.add_argument('--dataset',
                        help = 'Arquivo contendo dataset',
                        required=True
    )
    
    return parser.parse_args()




if __name__ == '__main__':
    main(parse_arguments())