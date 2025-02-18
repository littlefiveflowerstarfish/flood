import pandas as pd
import numpy as np
import os
from const import PATH
from utils import save_array_as_png
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from glob import glob

def prepare_df(tag):
    df = pd.read_csv(f'{PATH}/{tag}.csv')
    df['order'] = df['event_id'].apply(lambda x: x.split('_')[-1]).astype(int)
    df['location'] = df['event_id'].apply(lambda x: '_'.join(x.split('_')[:2]))
    return df

def prepare_data(tag):
    df = prepare_df(tag)
    if 'label' in df.columns:
        df = df.groupby('location').agg({'label': 'max'}).reset_index()
    else:
        df = df.drop_duplicates(subset=['location'])
    print(df.head())
    images_path = os.path.join(PATH, 'composite_images.npz')
    images = np.load(images_path)
    return df, images

def create_b3_images(tag, output='b3_images'):
    output = os.path.join(PATH, output)
    os.makedirs(output, exist_ok=True)

    df, images = prepare_data(tag)

    ids = df.location.values
    for id in tqdm(ids, total=len(ids)):
        image = images[id]
        save_array_as_png(image[:,:,3:], os.path.join(output, f'{id}.png'))

def create_b6_images(tag, output='b6_images'):
    output = os.path.join(PATH, output)
    os.makedirs(output, exist_ok=True)

    df, images = prepare_data(tag)

    ids = df.location.values
    for id in tqdm(ids, total=len(ids)):
        image = images[id]
        save_array_as_png(image[:,:,3:], os.path.join(output, f'{id}.png'))

def create_cv(img_folder='b3_images', output='b3_cv'):
    df = pd.read_csv(f'{PATH}/Train.csv')
    df['order'] = df['event_id'].apply(lambda x: x.split('_')[-1]).astype(int)
    df['location'] = df['event_id'].apply(lambda x: '_'.join(x.split('_')[:2]))
    df = df.groupby('location').agg({'label': 'max'}).reset_index()

    kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(df, df.label)):
        train = df.iloc[train_idx]
        val = df.iloc[val_idx]
        os.makedirs(f'{PATH}/{output}/fold_{fold}/train/flood', exist_ok=True)
        os.makedirs(f'{PATH}/{output}/fold_{fold}/train/no_flood', exist_ok=True)
        os.makedirs(f'{PATH}/{output}/fold_{fold}/val/flood', exist_ok=True)
        os.makedirs(f'{PATH}/{output}/fold_{fold}/val/no_flood', exist_ok=True)

        mask = train.label>0
        for i in train[mask].location.values:
            os.system(f'ln -s {PATH}/{img_folder}/{i}.png {PATH}/{output}/fold_{fold}/train/flood/{i}.png')
        for i in train[~mask].location.values:
            os.system(f'ln -s {PATH}/{img_folder}/{i}.png {PATH}/{output}/fold_{fold}/train/no_flood/{i}.png')
        
        mask = val.label>0
        for i in val[mask].location.values:
            os.system(f'ln -s {PATH}/{img_folder}/{i}.png {PATH}/{output}/fold_{fold}/val/flood/{i}.png')
        for i in val[~mask].location.values:
            os.system(f'ln -s {PATH}/{img_folder}/{i}.png {PATH}/{output}/fold_{fold}/val/no_flood/{i}.png')
    
   

if __name__ == '__main__':
    create_b3_images('Train')
    create_b3_images('Test')
    create_cv()