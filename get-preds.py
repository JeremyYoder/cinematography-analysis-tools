"""
Legacy backward-compatibility script for shot-type prediction.
This script now acts as a thin wrapper around the pip-installable package.
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='''
        ======================================================================
                 Predict shot types using a pretrained ResNet-50
        ======================================================================
        ''', formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--path_base', type=str, default=None,
                        help='path to the "shot-type-classifier" directory')
    parser.add_argument('--path_img', type=str, required=True,
                        help='path to where the images are stored')
    parser.add_argument('--path_preds', type=str, default=None,
                        help="path where you'd like to store the predictions")
    args = parser.parse_args()

    try:
        from cinematography_tools.predict import run_predictions
        run_predictions(
            path_base=args.path_base,
            path_img=args.path_img,
            path_preds=args.path_preds,
        )
    except ImportError:
        print("Error: cinematography_tools package not installed.")
        print("Please run: pip install -e .")
        import sys
        sys.exit(1)


if __name__ == '__main__':
    main()
