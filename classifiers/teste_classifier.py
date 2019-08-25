try:
    from utils.get_descriptor_images import *
    from utils.get_classifiers import *
    from utils.get_data import *
    from sklearn.model_selection import train_test_split
    from joblib import dump, load
    from sklearn.svm import SVC
    from sklearn.metrics import f1_score
    import numpy as np
    import pandas as pd

except Exception as e:
    print('Alguns módulos não foram instalados...')
    raise Exception(e)

def main():
    svm2 = load('classificador_svm.joblib')
    print(svm2)

    df_teste = pd.read_csv('lbp_features/01_TUMOR/1A11_CRC-Prim-HE-07_022.tif_Row_601_Col_151.csv')
    df_teste = np.array(df_teste).reshape(1, -1)
    y_pred = svm2.predict(df_teste)
    print(y_pred)
    y_pred = svm2.predict_proba(df_teste)
    print(np.round(y_pred, 3))

if __name__ == '__main__':

    main()