from __future__ import print_function, division
import os
import matplotlib
matplotlib.use('Agg')
import imageio
import torch
import numpy as np
from pathlib import Path
import pathlib
import argparse
from PIL import Image
import cv2
from utils.unet_resi_trans import UResnet_trans, BasicBlock, BottleNeck
from utils.general import (LOGGER,increment_path, check_suffix)

import warnings
import glob
warnings.filterwarnings("ignore")

use_gpu = True
Image.LOAD_TRUNCATED_IMAGES = True
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',default=ROOT / 'paper_image22')
    parser.add_argument('--mod', default='no_reference')
    parser.add_argument('--save-dir', default=ROOT / 'sample/paper_image21', help='dir of dataset')
    parser.add_argument('--weights', type=str, default=ROOT /'checkpoint/fine-euvp/fine-euvp-J.pt', help='initial weights path')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # --------------------------------
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=512, help='train, val image size (pixels)')
    # -------------------lr-------------------
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    cpu = opt.device == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif opt.device:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {opt.device} requested'  # check availability
    cuda = not cpu and torch.cuda.is_available()  #
    opt.device = torch.device("cuda:0" if cuda else "cpu")
    return opt

def load_data_val(path,mod):
    f =[]
    for p in path if isinstance(path, list) else [path]:
        p = Path(p)  # os-agnostic
        if p.is_dir():  # dir
            f += glob.glob(str(p / '**' / '*.*'), recursive=True)
            # f = list(p.rglob('*.*'))  # pathlib
        elif p.is_file():  # file
            with open(p) as t:
                t = t.read().strip().splitlines()
                parent = str(p.parent) + os.sep
                f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
        else:
            raise Exception(f'{p} does not exist')
    img_files=[]
    img_files += [x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS]
    Y = []
    if mod == 'reference':
        par =str( Path(Path(img_files[0]).parent).parent )+os.sep+'reference'
        Y += [par + os.sep +x.split(os.sep)[-1]  for x in img_files]
    elif mod=='no_reference':
        Y= img_files
    label_files = Y
    print(img_files[0:5])
    print(label_files[0:5])
    return img_files, label_files


if __name__ == "__main__":
    opt = parse_opt()

    pathlib.Path(opt.save_dir).mkdir(parents=True, exist_ok=True)
    check_suffix(opt.weights, '.pt')  # check weights
    pretrained = str(opt.weights).endswith('.pt')
    if pretrained:
        model_J = torch.load(opt.weights, map_location=opt.device)
    else:
        model_J = UResnet_trans(block=BasicBlock, layers=[3, 4, 6, 3], num_classes=3).to(opt.device)  # create

    psnr1,psnr2, ssim1,uiqm1, uciqe1,mse1=[],[], [], [], [],[]
    uiqm_m,uicm_m,uism_m,uiconm_m,uciqe_m, delta_m, con_m, mu_m=[],[],[],[],[],[],[],[]
    uiqm_m1, uicm_m1, uism_m1, uiconm_m1 = [],[],[],[]
    loss_train_path = opt.save_dir / 'psnr_ssim_uiqm.txt'
    with open(loss_train_path, 'a') as f:
        f.write(f'{opt.weights}')
        f.write(' ')
        f.write('\r\n')
        f.write(f'psnr')
        f.write(' ')
        f.write(f'ssim')
        f.write(' ')
        f.write(f'uiqm')
        f.write(' ')
        f.write(f'uciqe')
        f.write(' ')
        f.write(f'mse')
        f.write(' ')
        f.write('\r\n')
    img_files,label_files = load_data_val(opt.data_dir,opt.mod)
    opt.save_dir=str(opt.save_dir)
    idx=0
    for idx in range(0,len(img_files)):

        augment ,mosaic = False,False
        pad, stride= 0, 16
        img = cv2.imread(img_files[idx])
        ground = cv2.imread(label_files[idx])
        h0, w0 = img.shape[:2]
        r =400/ max(h0, w0)
        if r != 1:
           img = cv2.resize(img, (int(w0 * r), int(h0 * r)), # (), #int(w0 * r), int(h0 * r) #430 320cv2.INTER_AREA
                        interpolation=cv2.INTER_AREA if r < 1 and not augment else cv2.INTER_LINEAR)
           ground = cv2.resize(ground,(int(w0 * r), int(h0 * r)), #(256,256),
                        interpolation=cv2.INTER_AREA if r < 1 and not augment else cv2.INTER_LINEAR)
        h1, w1= img.shape[:2]
        r1= int(int(h1/stride)*stride)
        r2= int(int(w1/stride)*stride)
        if r1 != h1 or r2!=w1:
            img = img[0:r1, 0:r2, 0:3]
            ground = ground[0:r1, 0:r2, 0:3]
        img_name= (img_files[idx].split(os.sep)[-1]).split('.')[0]
        ground_name= (label_files[idx].split(os.sep)[-1]).split('.')[0]
        img =np.ascontiguousarray(img[:, :, ::-1].transpose((2, 0, 1)))
        img= torch.from_numpy(np.expand_dims(img,axis=0))
        img=img.to(opt.device, non_blocking=True).float() / 255
        ground =np.ascontiguousarray(ground[:, :, ::-1].transpose((2, 0, 1)))
        ground= torch.from_numpy(np.expand_dims(ground,axis=0))
        ground=ground.to(opt.device, non_blocking=True).float() / 255
        model_J.eval()
        output=model_J(img)
        f = opt.save_dir +os.sep+ f'{img_name}-P.png'
        output = output.clamp(0,1) #
        output = output.detach().cpu().numpy()
        output=np.squeeze(output).transpose((1, 2, 0))
        imageio.imsave(f, output*255)
