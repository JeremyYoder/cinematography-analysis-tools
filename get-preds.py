import argparse
import os
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore', '.*default behavior*', )
warnings.filterwarnings('ignore', '.*torch.solve*', )

def save_preds(path_img, learn, data, path_preds=None):
    from fastai.vision import open_image

    if path_preds is not None:
        os.makedirs(path_preds, exist_ok=True)

    cwd = os.getcwd()
    os.chdir(path_img)
    files = [f for f in os.listdir(path_img) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(files)

    preds_list = []

    for file in files:
        x = open_image(file)
        # get preds (probability of each class)
        preds_num = learn.predict(x)[2].numpy() * 100 # percentage
        preds_list.append([str(file)] + list(preds_num))

    os.chdir(cwd)

    if not preds_list:
        bdf = pd.DataFrame()
    else:
        # data.classes has the order of classes corresponding to preds_num
        cols = ['shot'] + list(data.classes)
        df = pd.DataFrame(preds_list, columns=cols)

        # tie-breaking hierarchy
        shot_order = ['LS', 'FS', 'MS', 'CS', 'ECS']

        # Filter available classes just in case
        available_shots = [s for s in shot_order if s in data.classes]

        if available_shots:
            # Get the top prediction
            df_preds = df[available_shots]
            # idxmax returns the first max, so if ordered LS, FS, MS, CS, ECS,
            # it will return LS in a tie.
            df['shot-type'] = df_preds.idxmax(axis=1)
            df['prediction'] = df_preds.max(axis=1)
            bdf = df[['shot-type', 'prediction', 'shot']].copy()
        else:
            bdf = pd.DataFrame()

    bdfname = "preds.csv"
    out_path = Path(path_preds) if path_preds is not None else Path(path_img)
    bdf.to_csv(out_path / bdfname, index=False)

if __name__ == '__main__':
    from initialise import get_model_data

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

    save_preds(path_img, learn, data, path_preds)
