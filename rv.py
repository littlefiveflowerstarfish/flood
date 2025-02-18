import warnings
warnings.filterwarnings('ignore')
from const import PATH
import os
import pandas as pd
from xgb_helper import XGBHelper
from sklearn.metrics import log_loss
from utils import gaussian_from_onehot
from functools import partial
import numpy as np

def prepare_data(tag='val'):
    if tag == 'val':
        df = pd.read_csv(f'{PATH}/Train.csv')
    else:
        df = pd.read_csv(f'{PATH}/Test.csv')
    print(df.head())
    df['order'] = df['event_id'].apply(lambda x: x.split('_')[-1]).astype(int)
    df['location'] = df['event_id'].apply(lambda x: '_'.join(x.split('_')[:2]))
    df['rsum'] = df.groupby('location')['precipitation'].transform('mean')
    df['rmax'] = df.groupby('location')['precipitation'].transform('max')
    df['days_since_rain'] = df.groupby('location')['precipitation'].transform(lambda x: (x == 0).cumsum() - (x == 0).cumsum().where(x != 0).ffill().fillna(0))
    rcols = []

    feas = ['precipitation', 'order', 'days_since_rain']

    models = ['x1']
    #df['m'] = 0
    for model in models:
        pdf = pd.read_csv(f'blend/{tag}_{model}_avg.csv')
        df = df.merge(pdf[['location','flood']], on='location', how='left')
        df = df.rename(columns={'flood': f'flood_{model}'})
        feas.append(f'flood_{model}')
    
    for w in range(2, 30):
        lag = w
        df[f'rainfall_lag_{lag}'] = df.groupby('location')['precipitation'].transform(lambda x: x.diff(lag).fillna(0))
        rcols.append(f'rainfall_lag_{lag}')

    for w in list(range(2,100,2)) + list(range(100, 250, 10)):
        df[f'rm_{w}'] = df.groupby('location')['precipitation'].transform(lambda x: x.rolling(w, center=True).mean())
        rcols.append(f'rm_{w}')


    for lag in [2,8,14,28]:
        for w in list(range(2, 150, 2)):
            df[f'lag_rm_{w}_{lag}'] = df.groupby('location')[f'rainfall_lag_{lag}'].transform(lambda x: x.rolling(w, center=True).mean())
            rcols.append(f'lag_rm_{w}_{lag}')
    
    for lag in [2,8,14,28]:
        for w in list(range(150,250,10)):
            df[f'lag_rm_{w}_{lag}'] = df.groupby('location')[f'rainfall_lag_{lag}'].transform(lambda x: x.rolling(w, center=True).mean())
            rcols.append(f'lag_rm_{w}_{lag}')

    feas += rcols
    return df, feas#+['m']

def cv(tag, sigma):
    df, feas = prepare_data()
    test, _ = prepare_data('test')
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
        m=1000
        tr['gaussian'] = tr.groupby('location')['label'].transform(partial(gaussian_from_onehot, sigma=sigma, m=m))
        val['gaussian'] = val.groupby('location')['label'].transform(partial(gaussian_from_onehot, sigma=sigma, m=m))

        xgb = XGBHelper('regression', params={'max_depth': 3, 'eta':0.02,
                                                  'subsample':0.7,
                                                  'min_child_weight':1,
                                                  'gamma':0.1,
                                                  'colsample_bytree':0.5}, 
                        num_boost_rounds=1000,
                        early_stop_rounds=100)
        e = 1e-5
        xgb.fit(tr[feas], tr['gaussian'], val[feas], val['gaussian'])
        importance = xgb.get_feature_importance()
        print(importance.head())
        df.loc[mask,'flood'] = np.clip(xgb.predict(val[feas])/m, e, 1-e)
        test['flood'] += np.clip(xgb.predict(test[feas])/m, e, 1-e)
        score = log_loss(df.loc[mask,'label'], df.loc[mask,'flood'])
        scores.append(score)
        print('fold', i, score)
        not_used = [i for i in feas if i not in importance.index.values.tolist()]
        #feas = [i for i in feas if i not in not_used]
        print('not used', not_used)
    score = log_loss(df['label'], df['flood'])
    
    print(scores)
    print('final', score)
    test['label'] = test['flood']/folds
    test[['event_id','label']].to_csv(f'blend/r_test_{tag}_{sigma}.csv', index=False, float_format='%.10f')
    df[['event_id','location','order','precipitation','label']+[i for i in df.columns if i.startswith('flood')]].to_csv(f'blend/r_val_{tag}_{sigma}.csv', index=False, float_format='%.10f')



if __name__ == '__main__':
    tag = 'x1'
    sigma = 15
    cv(tag, sigma)