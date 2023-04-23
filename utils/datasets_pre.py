from __future__ import print_function, division
import os
import pandas as pd
from scipy.io import loadmat
import numpy as np
import torch

from pathlib import Path
import cv2
import glob
from torch.utils.data import Dataset, DataLoader
from depth_estimate.model.pytorch_DIW_scratch import pytorch_DIW_scratch
import warnings
warnings.filterwarnings("ignore")
import random

use_gpu = True
from PIL import Image
###########################
from matplotlib import pyplot as plt
Image.LOAD_TRUNCATED_IMAGES = True

IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))  # DPP
NUM_THREADS = min(1, os.cpu_count())  # number of multiprocessing threads
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

def load_data(clean_file,depth_file, img_size=256, k_shots=10, k_query=10, ROOT_dir=None ):
    transformed_dataset_train = ImageRatingsDataset( clean_file,depth_file, img_size=img_size, batch_size=k_shots)
    transformed_dataset_valid = ImageRatingsDataset( clean_file,depth_file, img_size=img_size, batch_size=k_shots)
    dataloader_train = DataLoader(transformed_dataset_train, batch_size=k_shots, shuffle=True, num_workers=0, collate_fn=ImageRatingsDataset.my_collate)
    dataloader_valid = DataLoader(transformed_dataset_valid, batch_size=k_query, shuffle=True, num_workers=0,collate_fn=ImageRatingsDataset.my_collate)
    return dataloader_train, dataloader_valid

def load_imgfile(path):
    f = []
    path = path
    for p in path if isinstance(path, list) else [path]:
        p = Path(p)  # os-agnostic
        if p.is_dir():  # dir
            f += glob.glob(str(p / '**' / '*.*'), recursive=True)
        elif p.is_file():  # file
            with open(p) as t:
                t = t.read().strip().splitlines()
                parent = str(p.parent) + os.sep
                f += [x.replace('./', parent) if x.startswith('./') else x for x in t]
        else:
            raise Exception(f'{p} does not exist')
    file = []
    file += [x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS]
    return file

class ImageRatingsDataset(Dataset):
    """Images dataset."""
    def __init__(self,  clean_file,depth_file, img_size=640, batch_size=2, ROOT_dir=None):
        self.clean_data=load_imgfile(clean_file)
        random.shuffle(self.clean_data)
        self.depth_data = self.img2mat(self.clean_data,depth_file)
        self.img_size = img_size
        self.batch_size = batch_size

        self.depth = pytorch_DIW_scratch
        self.depth = torch.nn.parallel.DataParallel(self.depth, device_ids=[0])
        model_parameters = torch.load('./depth_estimate/checkpoints/test_local/best_generalization_net_G.pth')
        self.depth.load_state_dict(model_parameters)

    def __len__(self):
        return len(self.clean_data)

    def __getitem__(self, index):
        clean, depth= self.load_item(index)
        return clean, depth

    def load_item(self, index):

        clean = cv2.imread(self.clean_data[index])
        clean,xc,yc=self.get_square_img(clean)
        clean =cv2.resize(clean,(self.img_size,self.img_size) ,interpolation=cv2.INTER_CUBIC)

        if os.path.exists(self.depth_data[index]):
            depth =  np.array(loadmat(self.depth_data[index])['dph']).astype(np.float32)
            depth = self.get_square_refer(depth, xc, yc)
            depth = cv2.resize(depth, (self.img_size,self.img_size), interpolation=cv2.INTER_CUBIC)
        else:
            depth =self.depth_estimate(clean)
        """
        fig, ax = plt.subplots(1, 2, figsize=(6, 6))
        ax[0].imshow(clean[:, :, ::-1])
        ax[1].imshow(depth)
        plt.show()
        """
        clean = np.ascontiguousarray(clean[:, :, ::-1].transpose((2, 0, 1)))
        depth =np.ascontiguousarray( np.expand_dims(depth, axis= 0) )


        return torch.from_numpy(clean), torch.from_numpy(depth)

    @staticmethod
    def my_collate(batch):
        clean, depth = zip(*batch)
        return torch.stack(clean, 0), torch.stack(depth, 0)

    def depth_estimate(self,clean):
        clean1 = np.ascontiguousarray(clean[:, :, ::-1].transpose((2, 0, 1)))
        clean1 = torch.from_numpy(clean1).to(self.device, non_blocking=True).float() / 255
        clean1 = clean1.unsqueeze(0)
        self.depth.eval()
        with torch.no_grad():
            depth_hm = self.depth.forward(clean1)
            depth_hm = torch.exp(depth_hm)
            depth_hm = depth_hm.data.cpu().numpy()
            depth_hm =  np.squeeze(depth_hm)
        return depth_hm

    def get_square_img(self, img):
        h, w = img.shape[0:2]
        if h < w:
            x_b = random.randint(0, w- h)
            img= img[0:h,x_b:x_b+h,:]
            y_b=0
        elif h >= w:
            y_b = random.randint(0, h-w)
            img= img[y_b:y_b+h,0:h,:]
            x_b=0
        return img,x_b,y_b

    def get_square_refer(self, img,x_b,y_b):
        h, w = img.shape[0:2]
        if h < w:
            img=img[0:h,x_b:x_b+h]
        elif h >= w:
            img= img[y_b:y_b+h,0:h]
        return img

    def img2img(self, list, path):
        f = []
        f += [path + os.sep + str(Path(x).name) for x in list]
        return f

    def img2mat(self, list, path):
        f = []
        f += [path + os.sep + str(Path(x).stem) + '.mat' for x in list]
        return f

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im





