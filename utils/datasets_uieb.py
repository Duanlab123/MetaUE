from __future__ import print_function, division
import os
import numpy as np
from pathlib import Path
import cv2
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")
import random
import glob
use_gpu = True
from PIL import Image


Image.LOAD_TRUNCATED_IMAGES = True
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))  # DPP
NUM_THREADS = min(1, os.cpu_count())  # number of multiprocessing threads
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]


def load_data(mod='train', dataset='pretrained', worker_idx=0, img_size=256, k_shots=10, k_query=10, stride=16,
              hyp='None', ROOT_dir=None):
    data_dir = Path(ROOT_dir) / 'UIEB/Task1/underwater'
    transformed_dataset_train_1 = ImageRatingsDataset(mod='reference', csv_file=data_dir, root_dir=ROOT_dir,
                                                      img_size=img_size, batch_size=k_shots, stride=stride, hyp=hyp,
                                                      augment=False, rect=True, pad=0)
    transformed_dataset_valid_1 = ImageRatingsDataset(mod='reference', csv_file=data_dir, root_dir=ROOT_dir,
                                                      img_size=img_size, batch_size=k_shots, stride=stride, hyp=hyp,
                                                      augment=False, rect=True, pad=0)
    dataloader_train = DataLoader(transformed_dataset_train_1, batch_size=k_shots,
                                  shuffle=True, num_workers=0, collate_fn=ImageRatingsDataset.my_collate)
    dataloader_valid = DataLoader(transformed_dataset_valid_1, batch_size=k_query,
                                  shuffle=True, num_workers=0, collate_fn=ImageRatingsDataset.my_collate)

    return dataloader_train, dataloader_valid


class ImageRatingsDataset(Dataset):
    """Images dataset."""

    def __init__(self, csv_file, root_dir, img_size=640, batch_size=2, stride=16, hyp=None, augment=False, pad=0.0,
                 rect=False, mod='train'):
        # self.images_frame = pd.read_csv(csv_file, sep=',')
        self.root_dir = root_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        self.hyp = hyp
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.augment = augment
        self.rect = rect  #

        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)

        f = []
        path = csv_file
        for p in path if isinstance(path, list) else [path]:
            p = Path(p)  # os-agnostic
            if p.is_dir():  # dir
                f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                # f = list(p.rglob('*.*'))  # pathlib
            elif p.is_file():  # file
                with open(p) as t:
                    t = t.read().strip().splitlines()
                    parent = str(p.parent) + os.sep
                    f += sorted(
                        [x.replace('./', parent) if x.startswith('./') else x for x in t])  # local to global path
                    # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
            else:
                raise Exception(f'{p} does not exist')

        self.img_files = []
        self.img_files += [x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS]
        random.shuffle(self.img_files)
        assert self.img_files, f'No images found'
        ###############################################################cache#####################################
        Y = []
        if mod == 'reference':
            par = str(Path(Path(self.img_files[0]).parent).parent) + os.sep + 'reference'
            Y += [par + os.sep + x.split(os.sep)[-1] for x in self.img_files]
        elif mod == 'no_reference':
            Y = self.img_files
        self.label_files = Y

        # if mod == 'reference':
        #    par = str(Path(Path(self.img_files[0]).parent).parent) + os.sep + 'air_depth'
        #    Z += [par + os.sep + (x.split(os.sep)[-1]).split('.')[-2] + '.mat' for x in self.label_files]
        # elif mod == 'no_reference':
        Z = self.img_files
        self.air_depth = Z
        print(self.img_files[0:1])
        print(self.label_files[0:1])
        print(self.air_depth[0:1])
        #####################################################################################################

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        hyp = self.hyp
        mosaic = self.mosaic  # and random.random() < hyp['mosaic']
        self.indices = range(len(self.img_files))
        # mixup_idx = True
        if mosaic:
            s = self.img_size
            yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)
            indices = [idx] + random.choices(self.indices, k=3)
            random.shuffle(indices)
            for i, index in enumerate(indices):
                img = cv2.imread(self.img_files[index])
                ground = cv2.imread(self.label_files[index])
                h0, w0 = img.shape[:2]
                r = self.img_size / max(h0, w0)
                if r != 1:
                    im_train = cv2.resize(img, (int(w0 * r), int(h0 * r)),
                                          interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
                    ground_train = cv2.resize(ground, (int(w0 * r), int(h0 * r)),
                                              interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
                h, w = im_train.shape[:2]
                # place img in img4
                if i == 0:  # top left
                    img4 = np.full((s * 2, s * 2, im_train.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                    ground4 = np.full((s * 2, s * 2, ground_train.shape[2]), 114, dtype=np.uint8)
                    x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
                elif i == 1:  # top right
                    x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                    x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
                elif i == 2:  # bottom left
                    x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
                elif i == 3:  # bottom right
                    x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                    x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

                img4[y1a:y2a, x1a:x2a] = im_train[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
                ground4[y1a:y2a, x1a:x2a] = ground_train[y1b:y2b, x1b:x2b]
            image, ground = img4, ground4
            '''
            fig, ax = plt.subplots(1, 2, figsize=(6, 6))
            ax[0].imshow(img4[:, :, :])
            ax[1].imshow(ground4[:, :, ::-1])
            '''
        else:
            img = cv2.imread(self.img_files[idx])
            ground = cv2.imread(self.label_files[idx])
            if self.air_depth[idx].split('.')[-1].lower() in IMG_FORMATS:
                depth = cv2.imread(self.air_depth[idx])
            #else:
            #    depth = np.array(sio.loadmat(self.air_depth[idx])['dph']).astype(np.float32)

            # h0,w0=img.shape[:2]
            # h1,w1=ground.shape[:2]
            # if h0!=h1 or w0!=w1:
            #    img=img[0:min(h0,h1),0:min(w0,w1),:]
            #    ground=ground[0:min(h0,h1),0:min(w0,w1),:]
            #    depth=depth[0:min(h0,h1),0:min(w0,w1)]

            h0, w0 = img.shape[:2]
            r = self.img_size / max(h0, w0)
            if r != 1:
                img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
                ground = cv2.resize(ground, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
                depth = cv2.resize(depth, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)

            if depth.shape[-1] != 3:
                depth = np.expand_dims(depth, axis=2)
            '''
            fig, ax = plt.subplots(1, 3, figsize=(6, 6))
            ax[0].imshow(image[:, :, :])
            ax[1].imshow(ground[:, :, ::-1])
            ax[2].imshow(depth[:, :, 0])
            plt.show()
            '''
        return img, self.img_files[idx], ground, self.label_files[idx], depth, mosaic, [h0,
                                                                                        w0], self.img_size, self.stride, self.pad, self.augment

    @staticmethod
    def my_collate(batch):
        image, path_image, ground, path_ground, depth, mosaic, shapes, img_size, stride, pad, augment = zip(*batch)
        '''
        img_size, stride, pad, augment = img_size[0], stride[0], pad[0], augment[0]
        if not mosaic[0]:
            shapes=np.array(shapes)
            ar = shapes[:, 0] / shapes[:, 1]  # aspect ration
            shape_OPut = [1, 1]
            mini, maxi = ar.min(), ar.max()
            if maxi < 1:
                shape_OPut = [maxi,1]
            elif mini > 1:
                shape_OPut=[1,1 / mini]
            new_shape = np.ceil(np.array(shape_OPut) * img_size / stride + pad).astype(np.int) * stride
            ground_new, image_new = [], []
            image_new+=[letterbox(x, new_shape, auto=False, scaleup=augment) for x in image]
            ground_new+=[letterbox(x, new_shape, auto=False, scaleup=augment) for x in ground]

            fig, ax = plt.subplots(2, len(image_new), figsize=(6, 6))
            if len(image_new)==1:
                [ax[0].imshow(img[:, :, ::-1]) for index, img in enumerate(image_new)]
                [ax[1].imshow(ground[:, :, ::-1]) for index, ground in enumerate(ground_new)]
            else:
                [ax[0][index].imshow(img[:, :, ::-1]) for index,img in enumerate(image_new) ]
                [ax[1][index].imshow(ground[:, :, ::-1]) for index,ground  in enumerate(ground_new) ]

            image,ground = image_new, ground_new
            del image_new, ground_new

        ground_new, image_new=[] ,[]
        image_new+=[ np.ascontiguousarray(x[:, :, ::-1].transpose((2, 0, 1))) for x in image]
        ground_new+=[np.ascontiguousarray(x[:, :, ::-1].transpose((2, 0, 1))) for x in ground]
        ground_new1, image_new1=[] ,[]
        image_new1+=[torch.from_numpy(x) for x in image_new]
        ground_new1+=[torch.from_numpy(x) for x in ground_new]
        image=torch.stack(image_new1,0)
        ground=torch.stack(ground_new1,0)
        '''
        return image, path_image, ground, path_ground, depth, mosaic, shapes, img_size, stride, pad, augment


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


if __name__ == '__main__':
    train_path = 'E:/python-project/MetaIQA-underwater/pretrained/dic_test2.csv'
    image_path = 'E:/python-project/MetaIQA/tid2013.csv'
    imgsz = 256
    #    output_size = (imgsz, imgsz)
    #    transformed_dataset_train = ImageRatingsDataset(csv_file=train_path, root_dir=image_path,
    #                                                    transform=transforms.Compose(
    #                                                        [Rescale(output_size=(imgsz + 12, imgsz + 12)),
    #                                                         RandomHorizontalFlip(0.5),
    #                                                         RandomCrop(
    #                                                             output_size=output_size),
    #                                                         Normalize(),
    #                                                         ToTensor(),
    #                                                         ]))
    index = 30
    batch_size = 5
    hyp = {
        'fl_gamma': 0,  # (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
        'hsv_h': 0.015,  # (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
        'hsv_s': 0.7,  # (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
        'hsv_v': 0.4,  # (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
        'degrees': 0,  # (1, 0.0, 45.0),  # image rotation (+/- deg)
        'translate': 0.1,  # (1, 0.0, 0.9),  # image translation (+/- fraction)
        'scale': 0.5,  # (1, 0.0, 0.9),  # image scale (+/- gain)
        'shear': 0,  # (1, 0.0, 10.0),  # image shear (+/- deg)
        'perspective': 0,  # image perspective (+/- fraction), range 0-0.001
        'flipud': 0,  # image flip up-down (probability)
        'fliplr': 0.5,  # image flip left-right (probability)
        'mosaic': 1,  # image mixup (probability)
        'mixup': 1,  # image mixup (probability)
        'copy_paste': 1}  # segment copy-paste (probability)
    dataloader_train, dataloader_valid = load_data(mod='test', dataset='UIEB', worker_idx=1, img_size=imgsz,
                                                   batch_size=20, ROOT_dir=str(ROOT.parent), hyp=None)
    dataiter = iter(enumerate(dataloader_valid))
    #    for idx,(image,ground,path_image,path_ground) in enumerate(dataloader_train):
    #        t1=image
    for idx, (image, ground, path_image, path_ground) in enumerate(dataloader_valid):
        t1 = image

    dataloader_train, dataloader_valid = load_data(mod='train', dataset='pretrained', worker_idx=index, img_size=imgsz,
                                                   batch_size=batch_size, ROOT_dir=str(ROOT.parent), hyp=None)
    dataiter = iter(enumerate(dataloader_valid))
    #    for idx,(image,ground,path_image,path_ground) in enumerate(dataloader_train):
    #        t1=image
    for idx, (image, ground, path_image, path_ground) in enumerate(dataloader_valid):
        t1 = image
