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

def compute_days_since_last_rain(rain):
    """
    Compute the number of days since the last rain event for each day.
    
    Parameters:
    -----------
    rain : pd.Series
        A pandas Series with binary values, where 1 indicates a rain day and 0 indicates no rain.
    
    Returns:
    --------
    pd.Series
        A Series with the same index as `rain` where each value represents the number of days since the last rain.
        If there was no previous rain, the value will be NaN.
    """
    # Create an array of day indices.
    days = np.arange(len(rain))
    
    # Create a Series that holds the day index if it rained, otherwise NaN.
    last_rain_idx = pd.Series(np.where(rain == 1, days, np.nan), index=rain.index)
    
    # Forward-fill to propagate the most recent rain day index.
    last_rain_idx = last_rain_idx.ffill()
    
    # Calculate the difference between the current day and the last rain day.
    days_since_last_rain = days - last_rain_idx
    
    # For days before any rain occurred, set the value to NaN.
    days_since_last_rain[pd.isna(last_rain_idx)] = np.nan
    
    # Return the result as a pandas Series with a descriptive name.
    return pd.Series(days_since_last_rain, index=rain.index, name='days_since_last_rain')

def add_simple_features(df):
    new_features = []
    df['soil_moisture'] = df.groupby('location')['precipitation'].transform(lambda x: x.ewm(alpha=0.9).mean())
    new_features.append('soil_moisture')
    return df, new_features

from scipy.fft import fft

def add_all_features(df):
    """
    Adds all proposed features to the DataFrame and returns the DataFrame along with a list of new feature names.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with columns: 'location', 'precipitation', 'label', 'order'.
    
    Returns:
    --------
    df : pd.DataFrame
        DataFrame with all new features added.
    feas : list
        List of new feature names.
    """
    feas = []  # List to store new feature names
    
    # 1. Antecedent Precipitation Index (API)
    df['API'] = df.groupby('location')['precipitation'].transform(
        lambda x: x.ewm(alpha=0.1, adjust=False).mean()
    )
    feas.append('API')
    
    # 2. Rainfall Intensity Duration
    df['heavy_rain_3d'] = df.groupby('location')['precipitation'].transform(
        lambda x: x.rolling(3).sum() > 50  # 50mm threshold
    )
    feas.append('heavy_rain_3d')
    
    # 3. Dry-Wet Cycle Shock
    df['days_since_rain'] = df.groupby('location')['precipitation'].transform(
        lambda x: x.eq(0).cumsum().shift().fillna(0)
    )
    df['dry_then_wet'] = df['days_since_rain'] * df['precipitation']
    feas.extend(['days_since_rain', 'dry_then_wet'])
    
    # 4. Rolling Rainfall Variance
    df['rain_var_14d'] = df.groupby('location')['precipitation'].transform(
        lambda x: x.rolling(14).std()
    )
    feas.append('rain_var_14d')
    
    # 5. Flood Seasonality
    df['day_of_year'] = df['order'] % 365
    feas.append('day_of_year')
    
    # 6. Cumulative Rainfall Slope
    df['7d_rain_trend'] = df.groupby('location')['precipitation'].transform(
        lambda x: x.rolling(7).apply(lambda y: np.polyfit(range(7), y, 1)[0])
    )
    feas.append('7d_rain_trend')
    
    # 7. Consecutive Rain Days
    df['consec_rain'] = df.groupby('location')['precipitation'].transform(
        lambda x: x.gt(0).groupby((x.eq(0)).cumsum()).cumcount()
    )
    feas.append('consec_rain')
    

    
    # 9. Temporal Distance to Flood Peak
    def peak_distance(series):
        peak_day = series.idxmax()
        return series.index - peak_day
    df['days_from_peak'] = df.groupby('location')['precipitation'].transform(peak_distance)
    feas.append('days_from_peak')
    
    # 10. Spectral Features (FFT)
    df['fft_high_freq'] = df.groupby('location')['precipitation'].transform(
        lambda x: np.abs(fft(x))[5:10].mean()  # High frequency components
    )
    feas.append('fft_high_freq')
    
    # 11. Pressure Change Proxy
    df['pressure_change'] = df.groupby('location')['precipitation'].transform(
        lambda x: x.diff().rolling(3).mean()
    )
    feas.append('pressure_change')
    
    # 12. Basin Runoff Estimate
    df['runoff_est'] = df['API'] * df['precipitation'] ** 0.5
    feas.append('runoff_est')
    
    # 13. Event Co-Occurrence (example with wind data, if available)
    if 'wind_speed' in df.columns:
        df['wind_and_rain'] = (df['precipitation'] > 10) & (df['wind_speed'] > 15)
        feas.append('wind_and_rain')
    
    # 14. Spatial Neighbor Features (example with spatial data, if available)
    if 'latitude' in df.columns and 'longitude' in df.columns:
        from sklearn.neighbors import NearestNeighbors
        coords = df[['latitude', 'longitude']].values
        nbrs = NearestNeighbors(n_neighbors=5).fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        df['neighbor_7d_rain'] = df.groupby('location')['precipitation'].transform(
            lambda x: x.rolling(7).mean().iloc[indices].mean(axis=1)
        )
        feas.append('neighbor_7d_rain')
    
    # 15. Land Use Interaction (example with land cover data, if available)
    if 'urban_cover' in df.columns:
        df['urban_rain'] = df['precipitation'] * df['urban_cover']
        feas.append('urban_rain')
    
    return df, feas


def prepare_data(tag='val'):
    if tag == 'val':
        df = pd.read_csv(f'../notebooks/train_leak.csv')
    else:
        df = pd.read_csv(f'../notebooks/test_leak.csv')
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
    # tmp = df.groupby('location')['label'].max().reset_index()
    # mask = tmp.location.isin(df[df.flood_leak>0]['location'].unique())
    # assert tmp[mask]['label'].min()==1
    # assert tmp[~mask]['label'].max()==0

    models = ['a1','b1']

    for model in models:
        pdf = pd.read_csv(f'blend/{tag}_{model}_avg.csv')
        df = df.merge(pdf[['location','flood']], on='location', how='left')
        df = df.rename(columns={'flood': f'flood_{model}'})
        feas.append(f'flood_{model}')

    df['days_since_rain'] = df.groupby('location')['precipitation'].transform(compute_days_since_last_rain)
    df, cols = add_all_features(df)
    feas.extend(cols)


    for sigma in [50]:
        df[f'gaussian_{sigma}'] = df.groupby('location')['order'].transform(partial(gaussian_density, sigma=sigma))
        df[f'precipitation_{sigma}'] = df['precipitation']*df[f'gaussian_{sigma}']

    df['rsum'] = df.groupby('location')['precipitation'].transform('mean')
    df['rmax'] = df.groupby('location')['precipitation'].transform('max')
    rcols = []
    
    base = pd.read_csv('subs/cv_lb_0.002523834.csv').rename(columns={'label': 'base'})
    df = df.merge(base, on='event_id',how='left')
    df['base'] = fix(df,'base')
    df,rcols = fe(df, 'precipitation')
    #df,rcols1 = fe(df, 'days_since_rain')
    #df,rcols1 = fe(df, 'precipitation_50')
    

    feas += rcols# + rcols1
    return df, feas#+['base']

def cv(tag):
    df, feas = prepare_data()
    test, _ = prepare_data('test')
    feas = list(set(feas))
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
                                                  'scale_pos_weight': 2,
                                                  'gamma':0.1,
                                                  'colsample_bytree':0.5}, 
                        num_boost_rounds=5000,
                        early_stop_rounds=100)
        tr_margin = train_margin.loc[~mask, mcol].values
        val_margin = train_margin.loc[mask, mcol].values
        tr_margin,val_margin,test_margin = None,None,None

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