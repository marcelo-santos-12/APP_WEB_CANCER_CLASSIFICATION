try:
    import os
    import pandas as pd
    from wget import download
    from zipfile import ZipFile
    from glob import iglob
    from tqdm import tqdm
    from numpy import array, transpose
    
except Exception as e:
    print(e)
    raise Exception('Alguns módulos não foram instalados...')

def get_path_images(url, path_base):
    '''
    Rotina para busca da pasta com as imagens histológicas.
    Faz o download, extrai o zip e renomeia a pasta extraida para path_base, se a path_base não existir. 
    :url: str()
        link que aponta para o endereço do servidor que armazena as imagens.
    :path_base: st()
        pasta que contém as imagens.
    :Return: str()
        Nome da pasta onde as imagens estão guardadas.
    '''

    print('Iniciando busca da pasta de imagens...')
    filename = path_base
    if not os.path.exists(filename):
        print('Pasta ' + filename + ' não existe. Iniciando download...')
        try:
            filename_zip_images = download(url)
            print('\nDownload concluido...')
        except Exception as e:
            print(e)
            raise Exception('\nErro no download...')

        print('EXTRAINDO ZIP...')
        try:
            fantasy_zip = ZipFile(filename_zip_images)
            fantasy_zip.extractall('.')
            fantasy_zip.close()
            print('CONCLUINDO...')
        except:
            raise Exception('Erro ao extrair zip...')

        try:
            #renomeando pasta extraida
            os.system('mv ' + fantasy_zip.filename[:-4] + ' ' + filename)

        except:
            raise Exception('Erro ao renomear pasta...')
            
    else:
        print('Pasta encontrada...')

    return filename
    
def create_training_data(path_class, path_csv, verbose=True):
    '''
    Carrega os dados do descritor e o prepara para o treinamento.
    :path_class: str() 
        Path do descritor computados.
    :path_csv: str()
        Path onde será salvo o arquivos csv de treinamento.
    :Return: ndarray()
        Array da forma [n_sample, n_cols], [n_sample, class]
    '''
    assert os.path.exists(path_class) and isinstance(path_csv, str)
    CATEGORIES = os.listdir(path_class)

    dir_file_csv = os.path.join(path_class, path_csv + '.csv')

    if not os.path.exists(dir_file_csv):
        if verbose:
            print('Arquivo {} nao disponível'.format(path_csv))
            print('Iniciando criação do arquivo...')
        
        df_train = pd.DataFrame()
        
        for category in CATEGORIES: 
            path = os.path.join(path_class, category)  
            class_num = CATEGORIES.index(category)  # get number of the classification
            for csv in tqdm(iglob(path + '/*.csv')):

                try:
                    hog_array = pd.read_csv(csv)
                    hog_array = transpose(hog_array)   
                    
                    hog_serie = pd.Series(hog_array)

                    hog_serie.set_value(hog_array.shape[1], class_num)
                    
                    df_feature = pd.DataFrame(hog_serie)

                    df_train = pd.concat([df_train, df_feature])
                    
                except Exception as e: 
                    print(e)
                    raise Exception('Erro ao ler: {}'.format(csv))
        
        df_train.to_csv(dir_file_csv)
        x_train = df_train

    else:
        if verbose:
            print('Arquivo {} disponível...'.format(path_csv))
        x_train = pd.read_csv(dir_file_csv)

    x_train = array(x_train[:-1]).reshape(-1, x_train[0].shape[1])
    y_train = array(x_train[-1]).reshape(-1,)
    print(x_train.shape)
    print(y_train.shape)
    
    return x_train, y_train

def create_training_data(path_class, verbose=True):
    '''
    Carrega os dados do descritor e o prepara para o treinamento.
    :path_class: str() 
        Path do descritor computados.
    :path_csv: str()
        Path onde será salvo o arquivos csv de treinamento.
    :Return: ndarray()
        Array da forma [n_sample, n_cols], [n_sample, class]
    '''
    assert os.path.exists(path_class)

    CATEGORIES = os.listdir(path_class)
    
    if verbose:
        print('Iniciando organizacao do dataset...')
        
    x_train = []
    y_train = []
    for category in CATEGORIES:
        path = os.path.join(path_class, category)  
        class_num = CATEGORIES.index(category)  # get number of the classification
        for csv in tqdm(iglob(path + '/*.csv')):

            try:
                hog_array = array(pd.read_csv(csv))
                hog_array = transpose(hog_array)
                
                x_train.append(hog_array)
                y_train.append(class_num)

            except Exception as e: 
                print(e)
                raise Exception('Erro ao ler: {}'.format(csv))

    x_train = array(x_train).reshape(len(x_train), -1)
    y_train = array(y_train).reshape(-1,)
    print(x_train.shape)
    print(y_train.shape)
    
    return x_train, y_train