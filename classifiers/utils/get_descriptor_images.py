#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:20:14 2019

@author: marcelo
"""
try:
    from numpy import array, transpose
    import os
    import cv2
    import pandas as pd
    from threading import Thread
    from multiprocessing import Process
    from time import time
    from mahotas.features.lbp import lbp
    from skimage.feature import hog

except Exception as e:
    print(e)
    raise Exception('Alguns módulos não foram instalados...')

class LBP():
    def __init__(self, radius, points):
        self.radius = radius
        self.points = points

    def compute(self, image):
        return lbp(image, radius=self.radius, points=self.points, ignore_zeros=False)

    def __str__(self):
        return 'lbp'

class HOG():
    def __init__(self, ppc, cpb, orient):
        self.ppc = ppc
        self.cpb = cpb
        self.orient = orient
        
    def compute(self, image):
        hist = hog(image, orientations=self.orient, \
                        pixels_per_cell=self.ppc, cells_per_block=self.cpb, \
                        transform_sqrt=True, block_norm="L1")
        return hist

    def __str__(self):
        return 'hog'

def get_images(path, list_name_imgs):
    '''
    Cria um gerador contendo as imagens da pasta atual.
    :list_name_imgs: list(str())
        lista que contém os nomes das imagens a serem inseridas no gerador.
    :Return: generator()
        Gerador das imagens.
    '''
    return ([cv2.imread(os.path.join(path, name_img), cv2.IMREAD_GRAYSCALE), name_img] for name_img in list_name_imgs)

def thread_feature(DATADIR, OUTDIR, class_tissue, object_class, verbose=False):
    '''
    Computa os descritores de cada classe de tecidos.
    :DATADIR: str()
        Pasta que contém as imagens.
    :class_tissue: str()
        Classe atual para a qual serão computados os descritores.
    :Return:
        Sem retorno.
    '''
    if verbose:
        print('Iniciando execução na Categoria: ', class_tissue)
    
    path_class_imgs = os.path.join(DATADIR, class_tissue)

    list_name_imgs = os.listdir(path_class_imgs)
    
    imgs_generator = get_images(path_class_imgs, list_name_imgs) #gerador com todas as imagens da classe atual    
    
    for [img, name_img] in imgs_generator:
            
        path_feature = OUTDIR + '/' + class_tissue + '/'
    
        if not os.path.exists(path_feature):
            os.makedirs(path_feature)
    
        path_feature += name_img[:-4] + '.csv'
    
        feature = object_class.compute(img)
        df_feature = pd.DataFrame(feature)
        
        #SALVAR 
        df_feature.to_csv(path_feature, index=False)

    if verbose:
        print('Classe {} finalizada...'.format(class_tissue))

def create_features(DATADIR, OUTDIR, object_class, verbose=False,):
    '''
    Computa os descritores LBP's das imagens '.tif'.
    :DATADIR: str()
        Path que contem o diretório das imagens.
    :OUTDIR: st()
        Path onde serão armazenados os descritores LBP's.
    :Return:
        Sem retorno.
    '''

    feat = str(object_class)

    if verbose:
        print('Computando recursos {}´s...'.format(feat))      

    if os.path.exists(OUTDIR) and os.listdir(OUTDIR) != []:
        print('Descritores {}´s disponíveis...'.format(feat))
        return

    classes_tissues = os.listdir(DATADIR)
    classes_tissues.remove('08_EMPTY') #REMOVER CLASSE VAZIO

    threads = []
    for class_tissue in classes_tissues:
        t = Process(target=thread_feature, args=(DATADIR, OUTDIR, class_tissue, object_class, verbose,))
        t.start()
        
        init = time() 
        threads.append([t, init, class_tissue])

    return threads
