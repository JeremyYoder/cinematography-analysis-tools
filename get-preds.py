from initialise import *
import argparse
import warnings
import os
import pandas as pd
from pathlib import Path
from fastai.vision import open_image

warnings.filterwarnings('ignore', '.*default behavior*', )
warnings.filterwarnings('ignore', '.*torch.solve*', )

def save_preds(learn, data, path_img, path_preds=None):
    if path_preds is None:
        path_preds = path_img

    if not os.path.exists(path_preds):
        os.mkdir(path_preds)

    files = [f for f in os.listdir(
        path_img) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(files)

    preds_list = []

    # Pre-define the ordered categories for column re-ordering
    categories = ['LS', 'FS', 'MS', 'CS', 'ECS']

    for file in files:
        # open file
        x = open_image(os.path.join(path_img, file))

        # get preds
        preds_num = learn.predict(x)[2].numpy()

        preds_dict = {cat: pred for cat, pred in zip(data.classes, preds_num)}
        preds_dict['shot'] = str(file)
        preds_list.append(preds_dict)

    if preds_list:
        df = pd.DataFrame(preds_list)
        # Ensure only the expected categories are used for values, maintaining the tie-breaking sequence
        # Find highest prediction matching our categories sequence
        available_cats = [c for c in categories if c in df.columns]

        # Multiply by 100 to get percentage
        df[available_cats] = df[available_cats] * 100

        # Get the max value and the corresponding column
        df['prediction'] = df[available_cats].max(axis=1)
        df['shot-type'] = df[available_cats].idxmax(axis=1)

        bdf = df[['shot-type', 'prediction', 'shot']]
    else:
        bdf = pd.DataFrame(columns=['shot-type', 'prediction', 'shot'])

    bdfname = "preds.csv"
    bdf.to_csv(Path(path_preds)/bdfname, index=False)


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
