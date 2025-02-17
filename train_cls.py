# in ultralytics/utils/metrics.py
import os
import argparse
from const import PATH

parser = argparse.ArgumentParser(description="Train yolo detection model.")
# parser.add_argument("--yaml", type=str, required=True,
#                     help="yaml file path")
parser.add_argument('--model', type=str, help='pretrained model')
parser.add_argument('--tag', type=str, help='tag of config')
parser.add_argument('--gpu', type=str, default='0', help='gpu id to run inference')
parser.add_argument('--fold', type=int, help='fold id')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from ultralytics import YOLO

def train(model_name, tag, fold, save_dir):
    # model_dir = os.path.join('./save', model_name, f'fold_{fold}')
    # ckpt_path = os.path.join(model_dir, 'weights', "best.pt")
    model = YOLO(model_name)

    # Train the model
    model.train(
        data=os.path.join(PATH, "b6_cv", f'fold_{fold}'),
        epochs=10, 
        imgsz=128,
        batch=64,
        augment=True,
        mosaic=0,
        mixup=0,
        erasing=0,
        lr0=0.001,
        lrf=0.0001,
        #optimizer='adamw',
        flipud=0.5,
        auto_augment=None,
        project=save_dir,  # wandb project name
        name=tag+'/'+f'fold_{fold}',      # wandb run name
        )

if __name__ == '__main__':
    train(args.model, args.tag, args.fold, 'save')