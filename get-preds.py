import os
import argparse
import warnings
import pandas as pd
from pathlib import Path
from fastai.vision import open_image
from initialise import get_model_data

warnings.filterwarnings('ignore', '.*default behavior*', )
warnings.filterwarnings('ignore', '.*torch.solve*', )

def save_preds(path_img, path_preds=None, learn=None, data=None):
    if path_preds is not None:
        os.makedirs(path_preds, exist_ok=True)

    path_img_p = Path(path_img)
    files = [f for f in os.listdir(path_img) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(files)

    records = []
    if files and learn is not None and data is not None:
        for file in files:
            # open file
            x = open_image(path_img_p / file)

            # get preds
            preds_num = learn.predict(x)[2].numpy() * 100

            # form record
            records.append([str(file)] + list(preds_num))

    if records:
        df_all = pd.DataFrame(records, columns=['shot'] + list(data.classes))

        # order columns from largest to smallest shot size for tie-breaking
        order = [c for c in ['LS', 'FS', 'MS', 'CS', 'ECS'] if c in data.classes]
        df_preds = df_all[order]

        bdf = pd.DataFrame()
        bdf['shot-type'] = df_preds.idxmax(axis=1)
        bdf['prediction'] = df_preds.max(axis=1)
        bdf['shot'] = df_all['shot']
    else:
        bdf = pd.DataFrame(columns=['shot-type', 'prediction', 'shot'])

    bdfname = "preds.csv"
    if path_preds is not None:
        bdf.to_csv(Path(path_preds) / bdfname, index=False)
    else:
        bdf.to_csv(path_img_p / bdfname, index=False)

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

    parser.add_argument('--path_base', type=str, required=True,
                        help='path to the "shot-type-classifier" directory')
    parser.add_argument('--path_img', type=str, required=True,
                        help='path to where the images are stored')
    parser.add_argument('--path_preds', type=str, default=None,
                        help="path where you'd like to store the predictions")
    args = parser.parse_args()

    path = args.path_base
    path_img = args.path_img
    path_preds = args.path_preds

    learn, data = get_model_data(Path(path))
    learn = learn.to_fp32()

    save_preds(path_img, path_preds, learn, data)
