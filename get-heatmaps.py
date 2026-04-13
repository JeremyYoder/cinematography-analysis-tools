"""
Legacy backward-compatibility script for activation heatmaps.
This script now acts as a thin wrapper around the pip-installable package.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description='''
        ======================================================================
          Generate activation heatmaps of the ResNet-50 shot-type classifier
        ======================================================================
        ''', formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--path_base', type=str, default=None,
                        help='path to the "shot-type-classifier" directory')
    parser.add_argument('--path_img', type=str, required=True,
                        help='path to where the images are stored')
    parser.add_argument('--path_hms', type=str, default=None,
                        help="(optional) path where you'd like to store the heatmaps")
    parser.add_argument('--alpha', type=float, default=0.5,
                        help="degree to which you'd like to blend the heatmaps. Default = 0.5")
    args = parser.parse_args()

    try:
        from cinematography_tools.heatmap import generate_heatmaps
        generate_heatmaps(
            path_base=args.path_base,
            path_img=args.path_img,
            path_hms=args.path_hms,
            alpha=args.alpha,
        )
    except ImportError:
        print("Error: cinematography_tools package not installed.")
        print("Please run: pip install -e .")
        import sys
        sys.exit(1)


if __name__ == '__main__':
    main()
