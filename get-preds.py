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

    bdf_list = []

    shot_order = {s: i for i, s in enumerate(['LS', 'FS', 'MS', 'CS', 'ECS'])}

    for file in files:
        # open file
        x = open_image(file)

        # get preds
        preds_num = learn.predict(x)[2].numpy()

        # Combine predictions with classes and map to tie-breaking order
        preds = [(shot, float(prob) * 100, shot_order.get(shot, 99))
                 for shot, prob in zip(data.classes, preds_num)]

        # Sort by hierarchy index (tie-breaker) and then find the maximum by probability
        # In case of equal probabilities, max() returns the first one it encounters.
        # So we sort by tie-breaker first so the preferred shot type comes first.
        preds.sort(key=lambda x: x[2])
        best_shot = max(preds, key=lambda x: x[1])

        bdf_list.append({
            'shot-type': best_shot[0],
            'prediction': best_shot[1],
            'shot': str(file)
        })

    if bdf_list:
        bdf = pd.DataFrame(bdf_list)
    else:
        bdf = pd.DataFrame()

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
