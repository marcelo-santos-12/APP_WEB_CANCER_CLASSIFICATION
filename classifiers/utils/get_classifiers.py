#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Jul 25 20:15:17 2019

@author: marcelo
"""

try:
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    import pandas as pd

except Exception as e:
    print(e)
    raise Exception('Alguns módulos não foram instalados...')

def fit_grid_search(x_train, y_train, out):
    '''
    Realiza o treinamento de modelos SVM alterando os parâmetros.
    :x_train: ndarray()
        Amostras para treinamento da forma [n_sample, n_col].
    :y_train: ndarray()
        Classes das amostras de treinamento da forma [n_sample, class]
    :return: Sem retorno.
    '''

    param_grid = {
        'C': [10, 100, 200],
        'gamma': [0.001, 0.00001],
        'kernel': ['rbf', 'poly', 'linear', 'sigmoid'],
        'degree': [2, 3]
    }
    
    metrics = ['f1_macro', 'accuracy', 'precision_macro','recall_macro']

    grid_search = GridSearchCV(estimator=SVC(), \
                             param_grid=param_grid, \
                             scoring=metrics, \
                             refit=False, \
                             cv=5, \
                             n_jobs=-1,
                             verbose=100)

    grid_search.fit(x_train, y_train)
        
    print(100 * '.')
    print(grid_search.cv_results_)
    df_result = pd.DataFrame(grid_search.cv_results_)
    df_result.to_csv(out + '_resultados.csv')
    print(100 * '.')
