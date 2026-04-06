import os
import argparse
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from shutil import rmtree
from fastai.callbacks.hooks import hook_output
from matplotlib.ticker import NullLocator
from fastai.vision import Image, ImageDataBunch, ResizeMethod, imagenet_stats

from initialise import *

def hooked_backward(m, xb, y):
    # m[0] is the first part of the network i.e. NOT the FC layer
    with hook_output(m[0]) as hook_a:
        with hook_output(m[0], grad=True) as hook_g:
            preds = m(xb)
            preds[0,int(y)].backward()
    return hook_a,hook_g

def show_heatmap(xb_im, hm, path, y, idx, only_heatmap=False, interpolation='bilinear', alpha=0.5):
    _,ax = plt.subplots(figsize=(5,3))

    plt.gca().set_axis_off()
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())

    if not only_heatmap: xb_im.show(ax)
    ax.imshow(hm, alpha=alpha, extent=(0,666,375,0),
              interpolation=interpolation, cmap='YlOrRd')
    fname = f'{str(y)}_{str(idx+1)}_heatmap.png'
    plt.savefig(path/fname, bbox_inches = 'tight', pad_inches = 0, dpi=800)

    plt.close()
    plt.close('all')

def save_img(img, path, y, idx):
    img.show(figsize = (5,3))

    plt.gca().set_axis_off()
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())

    fname = f'{str(y)}_{str(idx+1)}.png'
    plt.savefig(path/fname, bbox_inches = 'tight', pad_inches = 0, dpi=800)

    plt.close()
    plt.close('all')

###############################################################################




###############################################################################
########################## GENERATING HEATMAPS ################################
###############################################################################


path_img = Path(path_img)
if path_hms is not None:
    path_hms = Path(path_hms)
else:
    path_hms = path_img

files = [f for f in os.listdir(path_img) if f.endswith(('.jpg', '.jpeg', '.png'))]

# creating the required directories where needed
# a dummy `ImageDataBunch` needs to be created to generate heatmaps
if path_hms is not None:
    os.mkdir(path_hms) if not os.path.exists(path_hms) else None

os.mkdir(path_img/'train') if not os.path.exists(path_img/'train') else None
os.mkdir(path_img/'train'/'img') if not os.path.exists(path_img/'train'/'img') else None

# move from base dir to dummy train dir
[os.rename(path_img/file, path_img/'train'/'img'/file) for file in files];


# dummy `ImageDataBunch`
temp = ImageDataBunch.from_folder(path_img, 'train', size = (375, 666), ds_tfms = None, bs=1,
                                  resize_method = ResizeMethod.SQUISH, no_check=True,
                                  num_workers = 0
                                 ).normalize(imagenet_stats)
# heatmap generation
for idx in range(len(temp.train_ds)):
    x,y = temp.train_ds[idx]
    print(f'# {idx+1} / {len(temp.train_ds)}')
    xb = temp.one_item(x)[0]
    if torch.cuda.is_available(): xb = xb.cuda()
    xb_im = Image(temp.denorm(xb)[0])
    hook_a,hook_g = hooked_backward()
    acts  = hook_a.stored[0].cpu()
    avg_acts = acts.mean(0)

    save_img(x, path_hms)
    show_heatmap(avg_acts, path_hms, only_heatmap=False, interpolation='spline16', alpha=alpha)


# deleting dummy directories and moving back files to where they were
[os.rename(path_img/'train'/'img'/file, path_img/file) for file in files];
rmtree(path_img/'train')
