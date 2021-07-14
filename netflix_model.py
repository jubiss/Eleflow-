import pandas as pd
import numpy as np
import sys
path = '/home/jubi/Documents/Pesquisa/Processo seletivo /Eleflow/Eleflow entregar'
sys.path.append(path)
import netflix_functions as nf

def model_results():
    #freq = range(10)
    final_test_recall = []
    final_test_f1 = []
    final_test_precision = []
    final_train_recall = []
    final_train_precision = []
    final_train_f1 = []
    
    
    enrich = False #Precisa do dataset para o enriquecimento
    percentile = 50
    n_calls = 10
    freq = 3
    variables = ['power listed_in','power director', 'power cast', 'power country','final_rating','size cast',
                 'size listed_in','size director','type','release_year','duration','years_since_release','year']
    if enrich == False:
            variables.remove('final_rating')
    #pip install openpyxl
    
    df = pd.read_excel(path+'/dataset_netflix.xlsx',engine='openpyxl')
    df = nf.preprocessing(df,enrich=enrich)
    
    from sklearn.model_selection import TimeSeriesSplit
    tcv = TimeSeriesSplit(n_splits=4)
    
    
    X, y = df, df['rating']
    
    recall_train = []
    recall_test = []
    precision_train = []
    precision_test = []
    f1_train = []
    f1_test = []
    
    
    for train_index, test_index in tcv.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
        X_train,y_train,X_test,y_test = nf.feature_eng_enc(X_train,y_train,X_test,y_test,freq,percentile=percentile)    
        X_train = X_train[variables]
        X_test = X_test[variables]
    
        from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
    
        model = nf.otimization(X_train,y_train,n_calls=n_calls)
    
        model.fit(X_train,y_train)
    
        pred_train = model.predict(X_train)
        
        recall_train.append(recall_score(y_train,pred_train))
        precision_train.append(precision_score(y_train,pred_train))
        f1_train.append(f1_score(y_train,pred_train))
    
        pred_test = model.predict(X_test)
        
        recall_test.append(recall_score(y_test,pred_test))
        precision_test.append(precision_score(y_test,pred_test))
        f1_test.append(f1_score(y_test,pred_test))
    
    last_conf = confusion_matrix(y_test,pred_test)
        
    final_test_recall.append(np.mean(recall_test))
    final_test_precision.append(np.mean(precision_test))
    final_test_f1.append(np.mean(f1_test))
    final_train_recall.append(np.mean(recall_train))
    final_train_precision.append(np.mean(precision_train))
    final_train_f1.append(np.mean(f1_train))

    return f1_test, f1_train, precision_test, precision_train, recall_test, recall_train, last_conf
