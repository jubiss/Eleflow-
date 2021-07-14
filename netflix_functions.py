import pandas as pd
import numpy as np

path = '/home/jubi/Documents/Pesquisa/Processo seletivo /Eleflow/'

def preprocessing(df,enrich=False):
    #Preprocessing
    df = df[(df['rating']!=0) & (df['rating'].isnull() == False)] 
    df['director'] = df.director.fillna('no director')
    df.dropna(inplace=True,subset=['date_added'])
    df.drop(['description'],axis=1,inplace=True)
    df['year'] = df['date_added'].dt.year
    df['years_since_release'] = df['year']-df['release_year']
    df['type'] = df['type'].apply(movie_tv)
  #  df['duration'] = df['duration'].apply(time)
    df = df[df['years_since_release']>=0]
    df = df.sort_values(by='date_added')
    if enrich==True:
            df = final_rating(df)
    df.reset_index(inplace=True)
    return df

def power(vec_genre, train_group_att, freq_max):
    '''
    vec_genre: Vetor com todos os gêneros do filme
    train_genre: Agrupamento dos valóres médios relacionados com cada genero (Data_set treino)
    min_freq: Número mínimo de filmes com mesmo genêro (Data_set_treino)
    '''
    power= 0
    num = 0
    for j in vec_genre:
        try:
            if train_group_att.loc[j]['show_id']>freq_max:
                num = num + 1
                power = train_group_att.loc[j]['rating'] + power
        except:
            return np.nan
    if power==0:
        return np.nan
    return power/num

def final_rating(df):
    df_rating = pd.read_csv(path+'rating.tsv',sep='\t')
    iter_csv = pd.read_csv(path+'movie_title.tsv',sep='\t', iterator=True, chunksize=1000)
    df_imdb = pd.concat([df_rating.join(chunk,how='inner') for chunk in iter_csv])
    df_imdb = df_imdb.set_index('title').join(df.set_index('title'),how='inner')
    
    df_imdb['total_rating'] = df_imdb['averageRating']*df_imdb['numVotes']
    df_imdb.reset_index(inplace=True)
    final_rating = df_imdb.groupby('title').agg({'total_rating':'sum','numVotes':'sum'})
    final_rating['final_rating'] = final_rating['total_rating']/final_rating['numVotes']
    
    df = df.set_index('title').join(final_rating['final_rating'],how='left')

    return df

def movie_tv(types):
    if types=='TV Show':
        return 0
    if types=='Movie':
        return 1

def time(minutagem):
    if type(minutagem) == str:
        if ' min' in minutagem:
            return int(minutagem.split(' min')[0])
    else:
        return np.nan
    
def size(a):
    if (a[0] == 'no director') or a[0] == 'none' or a[0] == 'bam':
       return 0
    else:
        return len(a)
    
    
def binary(a,q):
    if a>=q:
        return 0
    if a<q:
        return 1

def feature_eng_enc(X_train,y_train,X_test,y_test,freq,percentile):
    q = np.percentile(y_train,percentile)
    y_train = y_train.apply(binary,args=[q])
    y_test = y_test.apply(binary,args=[q])
    X_train['duration'] = X_train['duration'].apply(time)
    X_test['duration'] = X_test['duration'].apply(time)
    lista = ['listed_in','director','cast','country']
    for i in lista:
        X_train['single '+i] = X_train[i].str.split(",").apply(lambda x: [e.strip() for e in x]).tolist()
        X_train_g = X_train.explode('single '+i)
        X_train_g = X_train_g.groupby(['single '+i])[['rating','show_id']].agg({'rating':'mean','show_id':'nunique'})
        
        X_test['single '+i] = X_test[i].str.split(",").apply(lambda x: [e.strip() for e in x]).tolist()
        X_train['power '+i] = X_train['single '+i].apply(power, args=(X_train_g,freq))
        X_test['power '+i] = X_test['single '+i].apply(power, args=(X_train_g,freq))
        X_train['size '+i] = X_train['single '+i].apply(size)
        X_test['size '+i] = X_test['single '+i].apply(size)
    
    return X_train,y_train,X_test,y_test

def otimization(X_train,y_train,n_calls):
    space_xgb = [(1e-3, 1e-1, 'log-uniform'), # learning rate
              (100, 2000), # n_estimators
              (1, 10), # max_depth 
              (1, 4.), # min_child_weight 
              (0, 15), # gamma 
              (0.5, 1.), # subsample 
              (0.5, 1.)] # colsample_bytree 
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    import xgboost as xgb
    from skopt import dummy_minimize
    
    tcv_inner = TimeSeriesSplit(n_splits=2)
    def treina_xgb(params):
        learning_rate = params[0]
        n_estimators = params[1]
        max_depth = params[2]
        min_child_weight = params[3]
        gamma = params[4]    
        subsample = params[5]
        colsample_bytree = params[6]
        model = xgb.XGBClassifier(learning_rate=learning_rate,n_estimators=n_estimators,max_depth=max_depth,
                                  min_child_weight=min_child_weight, gamma=gamma, subsample=subsample,
                                  colsample_bytree=colsample_bytree)
        return -np.mean(cross_val_score(model,X_train,y_train,cv=tcv_inner,scoring="recall_weighted"))#mean_squared_error(y_test, p)
    
    resultado_xgb = dummy_minimize(treina_xgb,space_xgb,n_calls=n_calls,verbose=0)
    param_xgb = resultado_xgb.x

    model = xgb.XGBClassifier(learning_rate=param_xgb[0],n_estimators=param_xgb[1],max_depth=param_xgb[2],
                                  min_child_weight=param_xgb[3], gamma=param_xgb[4], 
                                  subsample=param_xgb[5],colsample_bytree=param_xgb[6])
    return model

    