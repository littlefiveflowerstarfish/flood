import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")
import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from PIL import Image
from utils import random_flip
from const import PATH
import yaml

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="yolo inference")

# Add arguments to the parser
parser.add_argument('--gpu', type=str, default='0', help='gpu id to run inference')
parser.add_argument('--data', type=str, choices=['test', 'val'], help='val or test')
parser.add_argument('--tag', type=str, help='tag of config')
parser.add_argument('--save_dir', type=str, default='./save', help='dir where weights are saved')
parser.add_argument('--data_dir', type=str, default=f'{PATH}/b3_images', help='dir where images live')
parser.add_argument('--fold', type=int, help='fold id')
parser.add_argument('--tte', type=int, default=0, help='tte repeat times')

# Parse the command-line arguments
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

def predict(test, model, aug, img_dir, tte=1):
    probs = []
    for _,row in tqdm(test.iterrows(), total=test.shape[0]):
        img_name = row['location']
        with Image.open(os.path.join(img_dir, img_name+'.png')) as img:
            pt = 0
            imgs = []
            for _ in range(tte):
                img = random_flip(img)
                imgs.append(img)
            if len(imgs)==0:
                imgs.append(img)
            res = model(imgs, verbose=False)
        for r in res:
            pt += r.probs.data.cpu().numpy()
        scale = 1.0/len(imgs)
        probs.append(pt*scale)
    probs = np.array(probs)
    print(res[0].names)
    for k,v in res[0].names.items():
        test[v] = probs[:,k]
    return test

if __name__ == '__main__':
    # Define the path to the downloaded model
    model_dir = os.path.join(args.save_dir, args.tag, f'fold_{args.fold}')
    ckpt_path = os.path.join(model_dir, 'weights', "best.pt")

    # check if model eixsts
    if not os.path.exists(ckpt_path):
        print('Model not found. Exiting...')
        exit()
    # check if task is classification
    yaml_path = os.path.join(model_dir, 'args.yaml')

    output = os.path.join(model_dir, f'{args.data}_res.csv')
    if os.path.exists(output):
        print('Output file already exists. Exiting...')
        exit()

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    aug = data['augment']
    print(aug)
    cls_model = YOLO(ckpt_path)

    if args.data == 'val':
        imgs = os.listdir(os.path.join(PATH, 'b3_cv', f'fold_{args.fold}', 'val', 'flood'))
        imgs += os.listdir(os.path.join(PATH, 'b3_cv', f'fold_{args.fold}', 'val', 'no_flood'))
        imgs = [i.split('.')[0] for i in imgs if i.endswith('.png')]
        test = pd.DataFrame(imgs, columns=['location'])
    elif args.data == 'test':
        test = pd.read_csv(os.path.join(PATH, 'Test.csv'))
        test['location'] = test['event_id'].apply(lambda x: '_'.join(x.split('_')[:2]))
        test = test.drop_duplicates(subset=['location'])
    else:
        assert 0, 'Invalid data type'
    # test = pd.read_csv(os.path.join(args.data_dir, f'fold_{args.fold}', 'val.csv'))
    img_dir = args.data_dir
    print(img_dir)

    sub = predict(test, cls_model, aug, img_dir, args.tte)
    sub.to_csv(output, index=False)