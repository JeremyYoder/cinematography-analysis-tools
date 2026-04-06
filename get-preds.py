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

def validate_path(p, check_exists=False):
    if p is not None:
        path_obj = Path(p).resolve()
        if '..' in Path(p).parts:
            raise ValueError(f"Path traversal detected in path: {p}")
        if path_obj.parts == ('/',) or path_obj.parent == Path('/'):
            raise ValueError(f"Dangerous path detected: {p}")
        if check_exists and not path_obj.is_dir():
            raise ValueError(f"Path does not exist or is not a directory: {p}")
    return p

def save_preds(learn, data, path_img, path_preds=None):
    if path_preds is not None:
        os.mkdir(path_preds) if not os.path.exists(path_preds) else None

    os.chdir(path_img)
    files = [f for f in os.listdir(
        path_img) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(files)

    bdf_list = []

    for file in files:
        # open file
        x = open_image(file)

        # get preds
        preds_num = learn.predict(x)[2].numpy()

        # form data-frame
        df = pd.DataFrame(list(zip(data.classes, preds_num)),
                          columns=['shot-type', 'prediction'])

        # reorder data-frame from largest to smallest shot size
        df['shot-type'] = pd.Categorical(df['shot-type'],
                                         ['LS', 'FS', 'MS', 'CS', 'ECS'])
        df = df.sort_values('shot-type').reset_index(drop=True)

        # probability --> percentage
        df['prediction'] *= 100

        df = df.sort_values('prediction', ascending=False)

        df = df.head(1)

        df['shot'] = str(file)

        bdf_list.append(df)

    if bdf_list:
        bdf = pd.concat(bdf_list, ignore_index=True)
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
    path_img = validate_path(args.path_img, check_exists=True)
    path_preds = validate_path(args.path_preds)

    learn, data = get_model_data(Path(path))
    learn = learn.to_fp32()

    save_preds(learn, data, path_img, path_preds)
