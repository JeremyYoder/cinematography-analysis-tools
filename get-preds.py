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

    # Optimization: To maximize performance, we avoid iterative pandas DataFrame
    # creation and sorting inside the loop. Instead, we use native Python operations
    # to find the best prediction per file and instantiate a single consolidated
    # DataFrame at the end.
    shot_type_order = {'LS': 0, 'FS': 1, 'MS': 2, 'CS': 3, 'ECS': 4}

    bdf_list = []

    for file in files:
        # open file
        x = open_image(file)

        # get preds
        # In environments where .numpy() output is mocked as standard Python lists,
        # we index it normally.
        preds_num = learn.predict(x)[2].numpy()

        # max() with tuple: primary key is prediction, secondary key is -order
        # to respect the tie-breaking hierarchy 'LS' > 'FS' > 'MS' > 'CS' > 'ECS'
        best_idx = max(range(len(data.classes)), key=lambda i: (preds_num[i], -shot_type_order.get(data.classes[i], 99)))

        bdf_list.append({
            'shot-type': data.classes[best_idx],
            'prediction': preds_num[best_idx] * 100.0,
            'shot': str(file)
        })

    if bdf_list:
        bdf = pd.DataFrame(bdf_list)
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
