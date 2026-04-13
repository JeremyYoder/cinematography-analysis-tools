import argparse
import os
import warnings

import pandas as pd
from fastai.vision import open_image
from pathlib import Path

from initialise import get_model_data

warnings.filterwarnings('ignore', '.*default behavior*', )
warnings.filterwarnings('ignore', '.*torch.solve*', )

def save_preds(learn, data, path_img, path_preds=None):
    if path_preds is not None:
        Path(path_preds).mkdir(parents=True, exist_ok=True)

    path_img_obj = Path(path_img)
    files = [f for f in path_img_obj.rglob('*') if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]

    print(f"Found {len(files)} images to process.")

    bdf_list = []
    hierarchy_map = {'LS': 0, 'FS': 1, 'MS': 2, 'CS': 3, 'ECS': 4}

    for idx, file in enumerate(files):
        print(f"Processing image {idx+1}/{len(files)}...")

        # open file
        x = open_image(file)

        # get preds
        preds_num = learn.predict(x)[2].numpy()

        # get best prediction, prioritizing probability then hierarchy
        preds = [(data.classes[i], float(preds_num[i]) * 100) for i in range(len(data.classes))]
        best_pred = max(preds, key=lambda p: (p[1], -hierarchy_map.get(p[0], 999)))

        bdf_list.append({
            'shot-type': best_pred[0],
            'prediction': best_pred[1],
            'shot': str(file.relative_to(path_img_obj))
        })

    if bdf_list:
        bdf = pd.DataFrame(bdf_list)
    else:
        bdf = pd.DataFrame(columns=['shot-type', 'prediction', 'shot'])

    bdfname = "preds.csv"
    if path_preds is not None:
        bdf.to_csv(Path(path_preds)/bdfname, index=False)
    else:
        bdf.to_csv(Path(path_img)/bdfname, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''
        ======================================================================
                 Predict shot types using a pretrained ResNet-50
        ======================================================================

         Usage
        -------

        python get-preds.py
            --path_base '/home/user/shot-type-classifier'
            --path_img '/home/user/Desktop/imgs'
            --path_preds '/home/user/Desktop/imgs/preds'
        ''', formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--path_base', type=str,
                        help='path to the "shot-type-classifier" directory')
    parser.add_argument('--path_img', type=str,
                        help='path to where the images are stored')
    parser.add_argument('--path_preds', type=str, default=None,
                        help="path where you'd like to store the predictions")
    args = parser.parse_args()

    path = args.path_base
    path_img = args.path_img
    path_preds = args.path_preds

    path_img_obj = Path(path_img)
    files = [f for f in path_img_obj.rglob('*') if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]

    if not files:
        print(f"No valid image files found in {path_img}")
    else:
        learn, data = get_model_data(Path(path))
        learn = learn.to_fp32()

        save_preds(learn, data, path_img, path_preds)
