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

    preds_list = []
    file_list = []

    for file in files:
        # open file
        x = open_image(file)

        # get preds
        preds_num = learn.predict(x)[2].numpy()
        preds_list.append(preds_num)
        file_list.append(str(file))

    if preds_list:
        classes = data.classes
        hierarchy = ['LS', 'FS', 'MS', 'CS', 'ECS']

        # Determine the correct tie-breaker column order
        ordered_cols = [c for c in hierarchy if c in classes] + [c for c in classes if c not in hierarchy]
        col_indices = [classes.index(c) for c in ordered_cols]

        # Order predictions to handle tie-breaking by hierarchy correctly
        preds_ordered = [[p[i] for i in col_indices] for p in preds_list]

        df_vec = pd.DataFrame(preds_ordered, columns=ordered_cols)

        # Vectorized ops for top prediction extraction
        bdf = pd.DataFrame({
            'shot-type': df_vec.idxmax(axis=1),
            'prediction': df_vec.max(axis=1) * 100,
            'shot': file_list
        })
        bdf['shot-type'] = pd.Categorical(bdf['shot-type'], categories=hierarchy)
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
