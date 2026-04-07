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
    path_img = validate_path(path_img, check_exists=True)

    if path_preds is not None:
        path_preds = validate_path(path_preds)
        os.mkdir(path_preds) if not os.path.exists(path_preds) else None

    os.chdir(path_img)
    files = [f for f in os.listdir(
        path_img) if f.endswith(('.jpg', '.jpeg', '.png'))]

    bdf_list = []
    hierarchy = ['LS', 'FS', 'MS', 'CS', 'ECS']

    for file in files:
        # open file
        x = open_image(file)

        # get preds
        preds_num = learn.predict(x)[2].numpy()

        # Determine best prediction efficiently using native python
        best_class_idx = -1
        best_prob = -1.0
        best_class = None

        for idx, (cls, prob) in enumerate(zip(data.classes, preds_num)):
            if prob > best_prob or (prob == best_prob and (best_class is None or hierarchy.index(cls) < hierarchy.index(best_class))):
                best_prob = prob
                best_class = cls
                best_class_idx = idx

        # Convert to percentage
        best_prob *= 100

        # Append as a dict
        bdf_list.append({
            'shot-type': best_class,
            'prediction': best_prob,
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

    path = validate_path(args.path_base, check_exists=True)
    path_img = args.path_img
    path_preds = args.path_preds

    learn, data = get_model_data(Path(path))
    learn = learn.to_fp32()

    save_preds(learn, data, path_img, path_preds)
