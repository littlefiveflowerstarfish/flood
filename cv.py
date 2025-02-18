import warnings
warnings.filterwarnings('ignore')
from const import PATH
import os
import pandas as pd
from xgb_helper import XGBHelper
from sklearn.metrics import log_loss
import numpy as np
from utils import gaussian_density
from functools import partial

def fix(df, col, confidence=0.05):
    if confidence is None:
        return df[col].values
    df['ls'] = df.groupby('location')[col].transform('sum')
    df['lx'] = df.groupby('location')[col].transform('max')
    mask = df.flood_a1 > confidence
    #mask = df.lx > confidence
    df.loc[mask,col] = df.loc[mask,col]/df.loc[mask,'ls']
    return df[col].values

def fe(df, col):
    rcols = []
    for w in list(range(2,100,2)) + list(range(100, 250, 10)):
        df[f'rm_{w}_{col}'] = df.groupby('location')[col].transform(lambda x: x.rolling(w, center=True).mean())
        rcols.append(f'rm_{w}_{col}')
    
    if col != 'precipitation':
        return df, rcols
    
    for w in range(2, 30):
        lag = w
        df[f'rainfall_lag_{lag}_{col}'] = df.groupby('location')[col].transform(lambda x: x.diff(lag).fillna(0))
        rcols.append(f'rainfall_lag_{lag}_{col}')


    for lag in [2,8,14,28]:
        for w in list(range(2, 150, 2)):
            df[f'lag_rm_{w}_{lag}_{col}'] = df.groupby('location')[f'rainfall_lag_{lag}_{col}'].transform(lambda x: x.rolling(w, center=True).mean())
            rcols.append(f'lag_rm_{w}_{lag}_{col}')
    
    for lag in [2,8,14,28]:
        for w in list(range(150,250,10)):
            df[f'lag_rm_{w}_{lag}_{col}'] = df.groupby('location')[f'rainfall_lag_{lag}_{col}'].transform(lambda x: x.rolling(w, center=True).mean())
            rcols.append(f'lag_rm_{w}_{lag}_{col}')
    return df, rcols

def prepare_data(tag='val'):
    if tag == 'val':
        df = pd.read_csv(f'{PATH}/Train.csv')
    else:
        df = pd.read_csv(f'{PATH}/Test.csv')
    feas = ['precipitation', 'order', 'days_since_rain']
    for r in [15]:
        dx = pd.read_csv(f'blend/r_{tag}_x1_{r}.csv')
        if tag == 'val':
            df[f'flood_{r}'] = np.clip(dx['flood'].values,0.001,1)
        else:
            df[f'flood_{r}'] = np.clip(dx['label'].values,0.001,1)
    print(df.head())

    df['order'] = df['event_id'].apply(lambda x: x.split('_')[-1]).astype(int)
    mi,mx = 100,600
    mask = (df['order'] < mi) | (df['order'] > mx)
    df.loc[mask,'precipitation'] = 0

    df['location'] = df['event_id'].apply(lambda x: '_'.join(x.split('_')[:2]))
    models = ['a1']

    for model in models:
        pdf = pd.read_csv(f'blend/{tag}_{model}_avg.csv')
        df = df.merge(pdf[['location','flood']], on='location', how='left')
        df = df.rename(columns={'flood': f'flood_{model}'})
        feas.append(f'flood_{model}')

    df['days_since_rain'] = df.groupby('location')['precipitation'].transform(lambda x: (x == 0).cumsum() - (x == 0).cumsum().where(x != 0).ffill().fillna(0))

    for sigma in [50]:
        df[f'gaussian_{sigma}'] = df.groupby('location')['order'].transform(partial(gaussian_density, sigma=sigma))
        df[f'precipitation_{sigma}'] = df['precipitation']*df[f'gaussian_{sigma}']

    df['rsum'] = df.groupby('location')['precipitation'].transform('mean')
    df['rmax'] = df.groupby('location')['precipitation'].transform('max')
    rcols = []
    
    
    df,rcols = fe(df, 'precipitation')
    
    feas += rcols
    return df, feas

def cv(tag):
    df, feas = prepare_data()
    test, _ = prepare_data('test')
    mcol = 'flood_15'
    train_margin = df[[mcol]]
    train_margin[mcol] = train_margin[mcol].values
    test_margin = test[mcol].values

    print(feas)
    folds = 4
    df['flood'] = 0
    test['flood'] = 0
    scores = []
    for i in range(folds):
        ids = os.listdir(os.path.join(PATH, 'b3_cv', f'fold_{i}/val','flood'))
        ids += os.listdir(os.path.join(PATH, 'b3_cv', f'fold_{i}/val','no_flood'))
        ids = [id.split('.')[0] for id in ids]
        print(len(ids))

        mask = df.location.isin(ids)
        tr = df[~mask]
        val = df[mask]
        print(tr.shape, val.shape)
        #print(tr.head())
        conf = 0.75

        xgb = XGBHelper('classification', params={'max_depth': 3, 'eta':0.02,
                                                  'subsample': 0.8,
                                                  'min_child_weight':1,
                                                  'gamma':0.1,
                                                  'colsample_bytree':0.5}, 
                        num_boost_rounds=5000,
                        early_stop_rounds=100)
        tr_margin = train_margin.loc[~mask, mcol].values
        val_margin = train_margin.loc[mask, mcol].values

        xgb.fit(tr[feas], tr['label'], val[feas], val['label'], tr_margin, val_margin)
        importance = xgb.get_feature_importance()
        print(importance.head())
        val['flood'] = xgb.predict(val[feas], val_margin)
        df.loc[mask,'flood'] = fix(val[['location','flood','flood_a1']], 'flood', conf)
        test['pred'] = xgb.predict(test[feas], test_margin)
        test['flood'] += fix(test[['location','pred','flood_a1']], 'pred', conf)
        score = log_loss(df.loc[mask,'label'], df.loc[mask,'flood'])
        scores.append(score)
        print('fold', i, score)
        not_used = [i for i in feas if i not in importance.index.values.tolist()]
        #feas = [i for i in feas if i not in not_used]
        print('not used', not_used)
    score = log_loss(df['label'], df['flood'])
    
    print(scores)
    print('final', score)
    test['label'] = test['flood'] / folds
    test[['event_id','label']].to_csv(f'blend/c_test_{tag}_avg.csv', index=False, float_format='%.10f')
    df[['event_id','location','order','precipitation','label']+[i for i in df.columns if i.startswith('flood')]].to_csv(f'blend/c_val_{tag}_avg.csv', index=False, float_format='%.10f')



if __name__ == '__main__':
    tag = 'x1'
    cv(tag)