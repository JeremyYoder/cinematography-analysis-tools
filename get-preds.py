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

    results = []
    hierarchy = {'LS': 0, 'FS': 1, 'MS': 2, 'CS': 3, 'ECS': 4}

    for file in files:
        # open file
        x = open_image(file)

        # get preds
        preds_num = learn.predict(x)[2].numpy()

        # find best prediction respecting the hierarchy for tie-breaks
        best_cls, best_pred = max(
            zip(data.classes, preds_num),
            key=lambda item: (item[1], -hierarchy.get(item[0], 99))
        )

        results.append({
            'shot-type': best_cls,
            'prediction': best_pred * 100,
            'shot': str(file)
        })

    if results:
        bdf = pd.DataFrame(results)
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
