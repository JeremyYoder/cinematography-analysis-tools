from initialise import *
import argparse
import warnings

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


def save_preds(path_img, path_preds=None):
    os.mkdir(path_preds) if not os.path.exists(path_preds) else None

    os.chdir(path_img)
    files = [f for f in os.listdir(
        path_img) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(files)

    bdf = pd.DataFrame()

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

        bdf = bdf.append(df, ignore_index=True)

        # save to disk
        #fname = file.rpartition('.')[0] + '_preds.csv'
        # if path_preds is not None:
        #    df.to_csv(Path(path_preds)/fname, index=False)

        # else:
        #    df.to_csv(Path(path_img)/fname, index=False)
    bdfname = "preds.csv"
    bdf.to_csv(Path(path_preds)/bdfname, index=False)


save_preds(path_img, path_preds)
