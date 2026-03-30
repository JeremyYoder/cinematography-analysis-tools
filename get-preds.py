import os
import argparse
import warnings
import pandas as pd
from pathlib import Path
from initialise import *

def save_preds(path_img, path_preds=None, learn=None, data=None):
    if path_img is None:
        return

    if path_preds is not None:
        os.mkdir(path_preds) if not os.path.exists(path_preds) else None

    # Get the original directory to restore it later
    orig_dir = os.getcwd()
    try:
        os.chdir(path_img)
        files = [f for f in os.listdir(
            path_img) if f.endswith(('.jpg', '.jpeg', '.png'))]
        print(files)

        if not files:
            bdf = pd.DataFrame()
        else:
            records = []
            for file in files:
                # open file
                from fastai.vision import open_image
                x = open_image(file)

                # get preds
                preds_num = learn.predict(x)[2].numpy() * 100

                record = dict(zip(data.classes, preds_num))
                record['shot'] = str(file)
                records.append(record)

            df = pd.DataFrame(records)

            # Enforce tie-breaking sequence
            classes_ordered = ['LS', 'FS', 'MS', 'CS', 'ECS']
            df_classes = df[classes_ordered]

            bdf = pd.DataFrame({
                'shot-type': df_classes.idxmax(axis=1),
                'prediction': df_classes.max(axis=1),
                'shot': df['shot']
            })

        bdfname = "preds.csv"
        if path_preds is not None:
            bdf.to_csv(Path(path_preds)/bdfname, index=False)
        else:
            bdf.to_csv(Path(path_img)/bdfname, index=False)
    finally:
        os.chdir(orig_dir)

if __name__ == '__main__':
    warnings.filterwarnings('ignore', '.*default behavior*', )
    warnings.filterwarnings('ignore', '.*torch.solve*', )

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

    save_preds(path_img, path_preds, learn=learn, data=data)
