import argparse
import pandas as pd
import numpy as np
import neuralnetwork




def main(arglist):

    #NORMALIZAR ATRIBUTOS
    if(arglist.dataset.split('.')[1] == 'csv'):
        dataset = pd.read_csv(arglist.dataset, sep=';')
    else:
        dataset = pd.read_csv(arglist.dataset, sep='\t')

    #print(dataset)

    normalizedDataset = dataset.apply(lambda x: (x-x.min())/(x.max()-x.min()), axis=0)
    
    #print(normalizedDataset)

    #VALIDACÃO CRUZADA

    #rede neural
    #uma lista de vetores [nx1](sendo n o numero de neuronios na camada)
    #uma lista de vetores de pesos para a conexão entre cada camada

    neural = neuralnetwork.NeuralNetwork(np.array([13,4,3,1]), 2, 2)
    neural.train(normalizedDataset)



def parse_arguments():

    parser = argparse.ArgumentParser(prog = 'Trabalho 2: Rede Neural utilizando backpropagation',
                                    description='Desenvolvido por Arthur Böckmann e João Dick\n',
                                    formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--dataset',
                        help = 'Arquivo contendo dataset',
                        required=True
    )
    
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments())