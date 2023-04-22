from __future__ import print_function, division
import os
import matplotlib
#matplotlib.use('Agg')
import torch
import numpy as np
from pathlib import Path
import pathlib
import argparse
from PIL import Image
import math
import copy
import cv2
import torch.optim as optim
import glob
from matplotlib import pyplot as plt
from torchvision.utils import save_image
#####################our function################

from utils.unet_resi_trans import UResnet_trans, BasicBlock
from utils.general import  check_suffix
from utils.datasets_pre import load_data, letterbox
from utils.uw_simulation import  wc_generator
###########################
import warnings

warnings.filterwarnings("ignore")
import random

use_gpu = True
Image.LOAD_TRUNCATED_IMAGES = True
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory


def load_data_val(path, mod):
    f = []
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
    img_files = []
    img_files += [x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS]
    Y = []
    if mod == 'reference':
        par = str(Path(Path(img_files[0]).parent).parent) + os.sep + 'reference'
        Y += [par + os.sep + x.split(os.sep)[-1] for x in img_files]
    elif mod == 'no_reference':
        Y = img_files
    label_files = Y
    print(img_files[0:5])
    print(label_files[0:5])
    return img_files, label_files


class Meta_UW:
    def __init__(self, opt, device):
        self.opt = opt
        self.save_dir = Path(opt.save_dir)
        self.device = device
        # ---------------------save task----------
        self.train_result = self.save_dir / 'weights'  # weights dir save_dir = ROOT/runs/train/exp/weights
        pathlib.Path(self.train_result).mkdir(parents=True, exist_ok=True)
        self.epoch_loss_path = self.save_dir / 'epoch_loss.txt'
        with open(self.epoch_loss_path, 'a') as f:
            f.write(f'{opt.weights, opt.weights_A, opt.weights_trans}')
            f.write(' ')
            f.write(f'epoch_loss_L_MM')
            f.write(' ')
            f.write(f'epoch_loss_L_trans')
            f.write(' ')
            f.write('\r\n')

    def train_model(self):
        check_suffix(opt.weights, '.pt')  # check weights
        pretrained = str(opt.weights).endswith('.pt')
        if pretrained:
            self.model_J = torch.load(opt.weights, map_location=self.device)
        else:
            self.model_J = UResnet_trans(block=BasicBlock, layers=[3, 4, 6, 3], num_classes=3).to(self.device)  # create
        check_suffix(opt.weights_A, '.pt')  # check weights
        pretrained = str(opt.weights_A, ).endswith('.pt')
        if pretrained:
            self.model_A = torch.load(opt.weights_A, map_location=self.device)
        else:
            self.model_A = UResnet_trans(block=BasicBlock, layers=[3, 4, 6, 3], num_classes=3).to(self.device)  # create
        check_suffix(opt.weights_trans, '.pt')  # check weights
        pretrained = str(opt.weights_trans).endswith('.pt')
        if pretrained:
            self.model_Trans = torch.load(opt.weights_trans, map_location=self.device)  # load checkpoint
        else:
            self.model_Trans = UResnet_trans(block=BasicBlock, layers=[3, 4, 6, 3], num_classes=3).to(
                self.device)  # create

        self.optimizer_J = optim.Adam(self.model_J.parameters(), lr=opt.alpha)
        self.optimizer_A = optim.Adam(self.model_A.parameters(), lr=opt.alpha)
        self.optimizer_Trans = optim.Adam(self.model_Trans.parameters(), lr=opt.alpha)

        self.model_J.cuda()
        self.model_A.cuda()
        self.model_Trans.cuda()

        self.meta_model_J = copy.deepcopy(self.model_J)
        self.temp_model_J = copy.deepcopy(self.model_J)
        self.meta_model_A = copy.deepcopy(self.model_A)
        self.temp_model_A = copy.deepcopy(self.model_A)
        self.meta_model_trans = copy.deepcopy(self.model_Trans)
        self.temp_model_trans = copy.deepcopy(self.model_Trans)

        self.psnr_path = self.save_dir / 'psnr.txt'
        with open(self.psnr_path, 'a') as f:
            f.write(f'{opt.weights, opt.weights_A, opt.weights_trans}')
            f.write(' ')
            f.write('\r\n')
            f.write(f'psnrs')
            f.write(' ')
            f.write('\r\n')

        self.epoch_loss=[]
        for epoch in range(0, opt.epochs):
            stop_nu = 400
            if epoch % stop_nu == 0:
                print(f'epoch', epoch)
                self.optimizer_J = exp_lr_scheduler(self.optimizer_J, epoch, lr_decay_epoch=stop_nu)
                self.optimizer_A = exp_lr_scheduler(self.optimizer_A, epoch, lr_decay_epoch=stop_nu)
                self.optimizer_Trans = exp_lr_scheduler(self.optimizer_Trans, epoch, lr_decay_epoch=stop_nu)
                decay_rate = 0.9 ** (epoch //stop_nu)

                self.learning = opt.learning * decay_rate
                self.alpha = opt.alpha * decay_rate
                print(f'learing {self.learning} alpha {self.alpha}')

            self.list_noise = list(range(1, 1 + opt.noise_num1))
            np.random.shuffle(self.list_noise)
            # ***************************random choose task_num to train*************
            # train_process : \theta_i = theta-\alpha\nabla L_i, theta= theta- \beta \sum_i L_i
            # -------------------some test--------------------------

            # -------------------some test--------------------------
            (x_qry,I_bar_val,y_qry,J_val, B_qry,A_val, t_qry,t_bar_val ) = self.train_process()
            # -------------------some test--------------------------
            # -------------------------------end task and start to finetunning----------------------------------------
            if epoch % stop_nu == 0:
                print('current loss = ', self.epoch_loss[-1])
                self.save_result('train_J', epoch,'J--->J-gt--->Input', J_val, y_qry, x_qry)
                self.save_result('train_B', epoch,'B--->B_gt--->I_Phy', A_val, B_qry, I_bar_val)
                self.save_result('train_trans', epoch,'t--->t_gt--->Input', t_bar_val, t_qry, x_qry)

            if epoch % stop_nu == 0:
                save_model = self.train_result / f'color J{epoch}.pt'
                epoch_model = copy.deepcopy(self.model_J)
                torch.save(epoch_model.cuda(), save_model)
                save_model = self.train_result / f'color A{epoch}.pt'
                epoch_model = copy.deepcopy(self.model_A)
                torch.save(epoch_model.cuda(), save_model)
                save_model = self.train_result / f'color Trans{epoch}.pt'
                epoch_model = copy.deepcopy(self.model_Trans)
                torch.save(epoch_model.cuda(), save_model)

        torch.save(self.model_Trans.cuda(), f'last_T.pt')
        torch.save(self.model_J.cuda(), f'last_J.pt')
        torch.save(self.model_A.cuda(), f'last_A.pt')

    def nkshot(self, temp_task=[1], k_shots=10, k_query=10):
        # --------------------prepare support set and query set, one task=[n_way*k_shots imgs]*3*256*256-----------------------------------------------------
        x_spt, y_spt, spt_trans,spt_back, x_qry, y_qry, qry_trans,qry_back = (), (), (), (), (), (),(),()
        # ---------------------------------------------------------------

        for index, num_noise in enumerate(temp_task):
            water_type = (num_noise - 1) % 10
            water_illum=np.random.uniform(0, 1,3)
            water_illum =water_illum/sum(water_illum)
            # ----------------------------------------------load_data(clean_file,depth_file, img_size=256, k_shots=10, k_query=10, ROOT_dir=None ):
            dataloader_train, dataloader_valid = load_data(opt.clean_dir,opt.depth_dir, opt.imgsz, k_shots=opt.k_shots,k_query=opt.k_query, ROOT_dir=str(ROOT))
            train_ground, depth = next( iter(dataloader_train))
            train_ground =train_ground.to(self.device, non_blocking=True).float()/255
            depth = depth.to(self.device, non_blocking=True).float()
            train_image, train_back, trans = wc_generator(train_ground, depth,
                                                          water_type=water_type,
                                                          water_depth=np.random.uniform(0.4, 10, 1),
                                                          water_illum=water_illum)
            x_spt += (train_image,)
            y_spt  +=(train_ground,)
            spt_trans+= (trans,)
            spt_back+=(train_back,)

            # ---------------------------------
            val_ground, val_depth= next( iter(dataloader_valid))
            val_ground =val_ground .to(self.device, non_blocking=True).float()/255
            val_depth = val_depth.to(self.device, non_blocking=True).float()
            val_image, val_back, val_trans = wc_generator(val_ground, val_depth,
                                                          water_type=water_type,
                                                          water_depth=np.random.uniform(0.4, 10, 1),
                                                          water_illum=water_illum)
            x_qry += (val_image,)
            y_qry += (val_ground,)
            qry_trans+=(val_trans,)
            qry_back+=(val_back,)

        x_spt = torch.cat(x_spt,dim=0)
        y_spt  = torch.cat(y_spt ,dim=0)
        spt_trans = torch.cat(spt_trans, dim=0)
        spt_back = torch.cat(spt_back,dim=0)
        #self.save_result( 'mod',1, 'title',  x_spt, y_spt,spt_trans)
        #self.save_result( 'mod', 1,'title',  spt_back, spt_back, spt_back)
        x_qry = torch.cat(x_qry,dim=0)
        y_qry  = torch.cat(y_qry ,dim=0)
        qry_trans = torch.cat(qry_trans, dim=0)
        qry_back = torch.cat(qry_back,dim=0)
        return x_spt, y_spt, spt_trans,spt_back, x_qry, y_qry, qry_trans,qry_back


    def train_process(self):
        name_to_param = dict(self.temp_model_A.named_parameters())
        for name, param in self.meta_model_A.named_parameters():
            diff = param.data - name_to_param[name].data
            name_to_param[name].data.add_(diff)

        name_to_param = dict(self.temp_model_J.named_parameters())
        for name, param in self.meta_model_J.named_parameters():
            diff = param.data - name_to_param[name].data
            name_to_param[name].data.add_(diff)

        name_to_param = dict(self.temp_model_trans.named_parameters())
        for name, param in self.meta_model_trans.named_parameters():
            diff = param.data - name_to_param[name].data
            name_to_param[name].data.add_(diff)

        running_loss, running_loss_trans = 0, 0

        self.model_J.train()
        self.model_A.train()
        self.model_Trans.train()
        for i in range(0, opt.task_num):# task_num=20
            temp_task = random.sample(self.list_noise, opt.n_ways)  # (n_ways*k_shots)*3*256*256

            x_spt, y_spt, t_spt, B_spt, x_qry, y_qry, t_qry,B_qry = self.nkshot(  temp_task=temp_task,k_shots=opt.k_shots, k_query=opt.k_query)
            # ------------------------------theta_i=theta-\alpha \nabla Loss-----------------------------------------------#
            name_to_param = dict(self.model_J.named_parameters())
            for name, param in self.temp_model_J.named_parameters():
                diff = param.data - name_to_param[name].data
                name_to_param[name].data.add_(diff)

            name_to_param = dict(self.model_A.named_parameters())
            for name, param in self.temp_model_A.named_parameters():
                diff = param.data - name_to_param[name].data
                name_to_param[name].data.add_(diff)

            name_to_param = dict(self.model_Trans.named_parameters())
            for name, param in self.temp_model_trans.named_parameters():
                diff = param.data - name_to_param[name].data
                name_to_param[name].data.add_(diff)

            # -------J = {Tensor}  Show Value--------------------------------
            J = self.model_J(x_spt)
            A = self.model_A(x_spt)
            t_bar = self.model_Trans(x_spt)

            J = J.clamp(0,1)
            A =   A.clamp(0.03,0.9)
            t_bar = t_bar.clamp(0.03,0.99)
            #I_tilde = torch.add(A, torch.mul( torch.sub(J,A),t_spt))
            I_bar =  torch.add(A, torch.mul( torch.sub(J,A),t_bar))
            self.optimizer_J.zero_grad()
            self.optimizer_A.zero_grad()
            self.optimizer_Trans.zero_grad()

            loss = self.model_Trans.L_MM_new(x_spt,I_bar,y_spt,J, B_spt,A, t_spt,t_bar, c1=1,c2=2,c3=1,c4=1)
            loss.backward()
            self.optimizer_J.step()
            self.optimizer_A.step()
            self.optimizer_Trans.step()
            # -----------------------------------valid the result-----------------------------------------------------
            J_val  = self.model_J(x_qry)
            A_val  = self.model_A(x_qry)
            t_bar_val = self.model_Trans(x_qry)
            J_val = J_val.clamp(0,1)
            A_val =   A_val.clamp(0.03,0.9)
            t_bar_val = t_bar_val.clamp(0.03,0.99)

            I_bar_val = torch.add(A_val, torch.mul(torch.sub(J_val, A_val), t_bar_val))

            self.optimizer_J.zero_grad()
            self.optimizer_A.zero_grad()
            self.optimizer_Trans.zero_grad()

            loss_val= self.model_Trans.L_MM_new(x_qry,I_bar_val,y_qry,J_val, B_qry,A_val, t_qry,t_bar_val, c1=1,c2=2,c3=1,c4=1)
            loss_val.backward()
            self.optimizer_J.step()
            self.optimizer_A.step()
            self.optimizer_Trans.step()

            running_loss += loss_val.item()
            running_loss_trans += loss_val.item()

            # -------------------------update para----------------------------------------------
            name_to_param1 = dict(self.meta_model_A.named_parameters())
            name_to_param2 = dict(self.temp_model_A.named_parameters())
            for name, param in self.model_A.named_parameters():
                diff = (param.data - name_to_param2[
                    name].data) / self.alpha  # theta_i^'-theta=theta-\alpha(\nabla f(\theta)+ \nabla f(\theta_i))-theta
                name_to_param1[name].data = name_to_param1[name].data + self.learning * (diff/opt.task_num)

            name_to_param1 = dict(self.meta_model_J.named_parameters())
            name_to_param2 = dict(self.temp_model_J.named_parameters())
            for name, param in self.model_J.named_parameters():
                diff = (param.data - name_to_param2[
                    name].data) / self.alpha  # theta_i^'-theta=theta-\alpha(\nabla f(\theta)+ \nabla f(\theta_i))-theta
                name_to_param1[name].data = name_to_param1[name].data + self.learning * (diff / opt.task_num)
                # name_to_param1[name].data.add_(diff / self.task_num)
            # -------------------------update para----------------------------------------------
            name_to_param1 = dict(self.meta_model_trans.named_parameters())
            name_to_param2 = dict(self.temp_model_trans.named_parameters())
            for name, param in self.model_Trans.named_parameters():
                diff = (param.data - name_to_param2[
                    name].data) / self.alpha  # theta_i^'-theta=theta-\alpha(\nabla f(\theta)+ \nabla f(\theta_i))-theta
                name_to_param1[name].data = name_to_param1[name].data + self.learning * (diff/opt.task_num)

            # -----------------------------------------------
        self.epoch_loss.append(running_loss / opt.task_num)
        with open(self.epoch_loss_path, 'a') as f:
            f.write(str(self.epoch_loss[-1]))
            f.write(' ')
            f.write('\r\n')

        return  x_qry,I_bar_val,y_qry,J_val, B_qry,A_val, t_qry,t_bar_val



        # ----------------------------end- finetunning------------------------------------------

    def norm_01(self, input):
        input_1 = []
        input_1 += [(input[i] - input[i].min()) / (input[i].max() - input[i].min()) for i in
                    range(input.shape[0])]
        return torch.stack(input_1, 0)

    def save_result(self, mod, epoch, title, outputs, x_qry, y_qry):
        # -------------------------------end finetunning and save some results----------------------------------------

        temp_out = outputs
        temp_input = x_qry
        temp_label = y_qry

        temp_out = temp_out.detach().cpu().numpy()
        temp_label = temp_label.detach().cpu().numpy()
        temp_input = temp_input.detach().cpu().numpy()
        # psnr1 = self.calculate_psnr(temp_out[0], temp_label[0])
        num_img = 5 if len(temp_out) > 6 else len(temp_out)
        fig, ax = plt.subplots(num_img, 3, figsize=(6, 6))
        if len(temp_out) == 1:
            [ax[0].imshow(temp_out[i].transpose((1, 2, 0))) for i, _ in enumerate(temp_out[0: num_img])]
            [ax[1].imshow(temp_input[i].transpose((1, 2, 0))) for i, _ in enumerate(temp_input[0: num_img])]
            [ax[2].imshow(temp_label[i].transpose((1, 2, 0))) for i, _ in enumerate(temp_label[0: num_img])]
        else:
            [ax[i][0].imshow(temp_out[i].transpose((1, 2, 0))) for i, _ in enumerate(temp_out[0: num_img])]
            [ax[i][1].imshow(temp_input[i].transpose((1, 2, 0))) for i, _ in enumerate(temp_input[0: num_img])]
            [ax[i][2].imshow(temp_label[i].transpose((1, 2, 0))) for i, _ in enumerate(temp_label[0: num_img])]

        f = self.train_result / f'{mod}_batch{epoch}_labels.png'
        plt.title(title)

        plt.savefig(f)
        plt.close()

    def valid_process(self, epoch, mod, dataset, temp_task, k_shots, k_query):
        # --------------end epoch and start valid------------------------------------------------
        # _,_, _,x_qry, y_qry, qry_depth = self.nkshot(mod=mod, dataset=dataset, temp_task=temp_task, k_shots=k_shots,
        #                                         k_query=k_query)
        # --------------------------finetunning-----------------------
        data_dir = './color/test/underwater'
        img_files, label_files = load_data_val(data_dir, 'reference')
        psnr_S, psnr_L = [], []
        for idx in range(0, len(img_files)):
            pad, stride = 0, 16
            img = cv2.imread(img_files[idx])
            ground = cv2.imread(label_files[idx])
            h0, w0 = img.shape[:2]
            r = 256 / max(h0, w0)
            if r != 1:
                img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
                ground = cv2.resize(ground, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
            h1, w1 = img.shape[:2]
            r1 = int(int(h1 / stride) * stride)
            r2 = int(int(w1 / stride) * stride)
            if r1 != h1 or r2 != w1:
                img = img[0:r1, 0:r2, 0:3]
                ground = ground[0:r1, 0:r2, 0:3]
            img = np.ascontiguousarray(img[:, :, ::-1].transpose((2, 0, 1)))
            img = torch.from_numpy(np.expand_dims(img, axis=0))
            img = img.to(self.device, non_blocking=True).float() / 255
            ground = np.ascontiguousarray(ground[:, :, ::-1].transpose((2, 0, 1)))
            ground = torch.from_numpy(np.expand_dims(ground, axis=0))
            ground = ground.to(self.device, non_blocking=True).float() / 255
            self.model_A.eval()
            self.model_J.eval()
            self.model_Trans.eval()
            with torch.no_grad():
                J = self.model_J(img)
                A = self.model_A(img)
                T = self.model_Trans(img)
            J = self.norm_01(J)
            A = self.norm_01(A)
            T = self.norm_01(T)
            img_name= Path(img_files[idx]).stem
            f = self.train_result / f'{img_name}_{3}.png'
            img_sample = torch.cat((img.data, J.data,ground.data), -1)
            save_image( img_sample, f, normalize=True)

            J = J.detach().cpu().numpy()
            ground = ground.detach().cpu().numpy()
            J = np.squeeze(J).transpose((1, 2, 0))
            ground = np.squeeze(ground).transpose((1, 2, 0))
            psnr1 = calculate_psnr(ground, J)
            psnr_S.append(psnr1)

        with open(self.psnr_path, 'a') as f:
            f.write(f'epoch')
            f.write(' ')
            f.write(f'psnr_s')
            f.write(' ')
            f.write('\r\n')
            f.write(f'{epoch}')
            f.write(' ')
            f.write(f'{round(sum(psnr_S) / len(psnr_S), 4)}')
        return J, img, ground, T, A


def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    # img1 = img1.squeeze()
    # img2 = img2.squeeze()

    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border]
    img2 = img2[border:h - border, border:w - border]

    img1 = (img1 * 255).astype(np.float64)
    img2 = (img2 * 255).astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255 / math.sqrt(mse))


def imsave(img, path):
    im = Image.fromarray(img.cpu().numpy().astype(np.uint8).squeeze())
    im.save(path)


def exp_lr_scheduler(optimizer, epoch, lr_decay_epoch=260):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""

    decay_rate = 0.9 ** (epoch // lr_decay_epoch)
    if epoch % lr_decay_epoch == 0:
        print('decay_rate is set to {}'.format(decay_rate))

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate
        print(f'learning rate', param_group['lr'])
    return optimizer


def parse_opt(known=False):
    #     opt.save_dir , opt.epochs, opt.batch_size, opt.weights, opt.task_num, opt.noise_num1, opt.path, opt.evolve
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean-dir', default= './dataset/clean', help='dir of dataset')
    parser.add_argument('--depth-dir', default='./dataset/depth', help='dir of dataset')

    parser.add_argument('--weights', type=str, default=   './checkpoint/', help='initial weights path')
    parser.add_argument('--weights_A', type=str, default=  './checkpoint/', help='initial weights path')
    parser.add_argument('--weights_trans', type=str, default =   './checkpoint/', help='initial weights path')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # --------------------------------
    parser.add_argument('--epochs', type=int, default=10001)
    parser.add_argument('--task-num', type=int, default=5, help='batch size for task')
    parser.add_argument('--n-ways', type=int, default=1, help='n-ways distorbute')
    parser.add_argument('--k-shots', type=int, default=2, help='one -way, k example to train')
    parser.add_argument('--k-query', type=int, default=2, help='one -way, k example to train')
    parser.add_argument('--noise-num1', type=int, default=30, help='total task for fisrt data set')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=128, help='train, val image size (pixels)')
    # -------------------lr-------------------
    parser.add_argument('--learning', default=1e-4,
                        help='learnging rate')  # learing rate for meta net, i.e. beta=1e-4, \theta = theta-beta\nabla\sum_i Loss_i
    parser.add_argument('--alpha', default=1e-4,
                        help='learnging rate')  # learing rate for meta net, i.e. beta=1e-4, \theta = theta-beta\nabla\sum_i Loss_i
    parser.add_argument('--learning_refine', default=1e-4, help='learnging rate')
    # -------------------save result-------------------
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp1', help='save to project/name')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt
if __name__ == "__main__":
    opt = parse_opt()
    opt.save_dir = str(Path(opt.project) / opt.name)
    cpu = opt.device == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif opt.device:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {opt.device} requested'  # check availability
    cuda = not cpu and torch.cuda.is_available()  #
    device = torch.device("cuda:0" if cuda else "cpu")
    meta_uw = Meta_UW(opt, device)
    meta_uw.train_model()
