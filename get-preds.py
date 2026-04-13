import os
import pandas as pd
from pathlib import Path
from fastai.vision import open_image
from initialise import *
import argparse
import warnings
import os
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore', '.*default behavior*', )
warnings.filterwarnings('ignore', '.*torch.solve*', )

def save_preds(learn, data, path_img, path_preds=None):
    if path_preds is not None:
        os.mkdir(path_preds) if not os.path.exists(path_preds) else None

    os.chdir(path_img)
    files = [f for f in os.listdir(
        path_img) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(files)

    records = []

    shot_hierarchy = ['LS', 'FS', 'MS', 'CS', 'ECS']

    for file in files:
        # open file
        x = open_image(file)

        # get preds
        preds_num = learn.predict(x)[2].numpy()

        # probability --> percentage
        preds_pct = preds_num * 100

        # map class to prediction value
        pred_dict = dict(zip(data.classes, preds_pct))

        # Select best shot type based on hierarchy ties using max
        best_shot = None
        best_pred = -1
        for shot_type in shot_hierarchy:
            if shot_type in pred_dict:
                if pred_dict[shot_type] > best_pred:
                    best_pred = pred_dict[shot_type]
                    best_shot = shot_type

        records.append({
            'shot-type': best_shot,
            'prediction': best_pred,
            'shot': str(file)
        })

    if records:
        bdf = pd.DataFrame(records)
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

    learn, data = get_model_data(Path(path))
    learn = learn.to_fp32()

    save_preds(learn, data, path_img, path_preds)
