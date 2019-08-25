#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:07:14 2019

@author: marcelo
"""
__version__ = '1.0'
__author__ = 'Marcelo dos Santos (mar.mhps@gmail.com)'

try:
    from utils.get_descriptor_images import *
    from utils.get_classifiers import *
    from utils.get_data import *
    from sklearn.model_selection import train_test_split
    from joblib import dump, load
    from sklearn.svm import SVC
    from sklearn.metrics import f1_score

except Exception as e:
    print('Alguns módulos não foram instalados...')
    raise Exception(e)

GAMMA = 1e-06
C = 10
KERNEL = 'rbf'

svm = SVC(kernel=KERNEL, gamma=GAMMA, C=C, probability=True)

def main():
    #------------------------------------------------------------------------------------------
    #BUSCA DA PASTA DE IMAGENS
    #link onde as imagens podem ser baixadas
    url = 'https://zenodo.org/record/53169/files/Kather_texture_2016_image_tiles_5000.zip?download=1'
    
    #diretorio onde elas serão guardadas após baixadas e extraida do zip
    path_base = 'tissue_class'
    
    #Rotina para busca da pasta com as imagens histológicas
    DATADIR = get_path_images(url, path_base)
    #------------------------------------------------------------------------------------------
    #COMPUTANDO DESCRITORES
    obj_lbp = LBP(radius=4, points=14)
    #computa os descritores
    OUTDIR = 'lbp_features'
    threads = create_features(DATADIR, OUTDIR, obj_lbp, verbose=True,)
    
    #------------------------------------------------------------------------------------------
    #TREINAMENTO
    
    print('Organizando dados de treinamento de \'/{}\'...'.format(OUTDIR))
    path_csv = OUTDIR
    
    x_data, y_data = create_training_data(OUTDIR, path_csv)
    
    print('Treinando SVM com {}...'.format(OUTDIR))

    print('Tamanho do vetor de entrada no SVM: ', x_data[0].shape)
    print('Formato da saida de classificação: ', y_data[0].shape)
    
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=50)

    print('x_train: ', x_train.shape)
    print('y_train: ', y_train.shape)
    svm.fit(x_train, y_train)

    #Salvando modelo
    dump(svm, 'classificador_svm.joblib')

    # para carregar o modelo, basta usar
    svm2 = load('classificador_svm.joblib')
    
    y_pred = svm2.predict(x_test)

    results = f1_score(y_test, y_pred, average='macro')

    print('F1 Score: ', results)
    print('Finalizado...')

if __name__ == '__main__':

    main()
