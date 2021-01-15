# -*- coding: utf-8 -*-

"""
Created on Wed Jul 14 20:15:17 2019

@author: marcelo
"""

try:
    import cv2
    import time
    from joblib import load
    from classifiers.utils.get_descriptor_images import LBP
    import numpy as np
except Exception as e:
    raise Exception(e)

HEIGHT_WINDOW_SLIDING = 150
WIDTH_WINDOW_SLIDING = 150

CLASSES_STR = ['COMPLEX', 'ADIPOSE', 'MUCOSA', 'TUMOR', 'STROMA', 'DEBRIS', 'LYMPHO']

def accepted_image(image, mean_pixel_limit, central_pixel):
    '''
    Verifica se a imagem é válida para servir como entrada no classificador.
    :image: imagem a ser verificada.
    :mean_pixel_limit: valor da média de pixel como limite da validação.
    :Return: bool indicando se é válida ou não.
    '''

    (ret, bin_img) = cv2.threshold(image, central_pixel, 255, cv2.THRESH_BINARY_INV)

    mean_pixel = bin_img.mean()
    print('Binary Limit: ', ret)
    print('Media dos pixels: ', mean_pixel)

    '''cv2.imshow('test', bin_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''

    if mean_pixel > mean_pixel_limit:
        return True 
    else:
        return False

def divide_image(image):
    '''
    Corta imagem em regiões iguais de altura HEIGHT_WINDOW_SLIDING, largura WIDTH_WINDOW_SLIDING e em passo stride.
    :image: ndarray(imagem de entrada)
    :Return: generator(Imagens cortadas)
    '''
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    heigth, width = img_gray.shape
    assert heigth%HEIGHT_WINDOW_SLIDING == 0 or width%WIDTH_WINDOW_SLIDING == 0
    return (img_gray[n_column : n_column + WIDTH_WINDOW_SLIDING, n_row:n_row + HEIGHT_WINDOW_SLIDING] \
        for n_column in range(0, width, WIDTH_WINDOW_SLIDING)  \
            for n_row in range(0, heigth, HEIGHT_WINDOW_SLIDING))

def image_classifier(image):
    '''
    Classifica a image carregada pelo usuário.
    :image_gen: generator(Imagens cortadas em dimensões iaguais para o classificador)
    :Return: str(classe a qual a imagem pertence).
    '''
    assert isinstance(image, np.ndarray)

    H, W, _ = image.shape
    H_NEW = HEIGHT_WINDOW_SLIDING * int(H/HEIGHT_WINDOW_SLIDING) # redimensiona a image de forma que as dimensoes sejam divisiveis por 150
    W_NEW = WIDTH_WINDOW_SLIDING * int(W/WIDTH_WINDOW_SLIDING)
    image = cv2.resize(image, (H_NEW, W_NEW))

    image_gen = divide_image(image)

    lbp = LBP(radius=4, points=14)
    svm = load('classificador_svm.joblib')
    
    classes = []

    for img in image_gen:
        #if accepted_image(img, mean_pixel_limit=50, central_pixel=150):
        descriptor_image = lbp.compute(img)
        #classes.append(svm.predict_proba(descriptor_image.reshape(1, -1)))
        classes = svm.predict_proba(descriptor_image.reshape(1, -1))
    
    return np.round(classes, 3)
    
    #Escolher classe que mais foi classificada
    classes_options, counts = np.unique(classes, return_counts=True)    
    
    if len(classes) > 0:
        qtd_max = np.max(counts)
        indice_max = list(counts).index(qtd_max)
        index_class_choiced = classes_options[indice_max]
        #return classes_str[index_class_choiced]
        return classes[index_class_choiced]
    
    else: #ou seja, se a imagem possuir as mesmas dimensoes da janela deslizante
        #return classes_str[classes[0]] #ou seja, se a imagem possuir as mesmas dimensoes da janela deslizante
        return classes[0]

def main():
    #img = cv2.imread('/home/marcelo/Downloads/Kather_texture_2016_larger_images_10/CRC-Prim-HE-01_APPLICATION.tif')
    img = cv2.imread('/home/marcelo/Documentos/LGHM/APP_WEB/tissue_class/01_TUMOR/10A26_CRC-Prim-HE-07_025.tif_Row_1801_Col_301.tif')

    classe = image_classifier(img)

    print('Classe escolhida: ', classe)

if __name__ == '__main__':

    main()
