import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import argparse
from tqdm import tqdm
from PIL import  Image
import scipy.io as sio
import imageio
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']


def load_image(data_path):
    f = []
    for p in data_path if isinstance(data_path, list) else [data_path]:
        p = Path(p)
        if p.is_dir():
            f += glob.glob(str(p / '*.*'), recursive=True)
        elif p.is_file():
            with open(p) as t:
                t = t.read().strip().splitlines()
                parent = str(p.parent) + os.sep
                f += [x.replace('./', parent) if x.startswith('./') else x for x in t]

    img_files = [x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS]
    return img_files
def png2mat(data_path,depth_path):
    depth=[]
    depth += [str(depth_path) +os.sep+ str(Path(x).stem)+'.mat' for x in data_path ]
    return depth
class un_generator(object):
    def __init__(self,batch_size=1, air_dir='default', depth_dir='default',results_dir=None):
        self.batch_size = batch_size
        self.air_dir = air_dir
        self.depth_dir = depth_dir
        self.results_dir = results_dir
        ####################step 1 : read image and depth.mat, and check it ###############################
        self.img_files = load_image(self.air_dir)
        self.depth_files = png2mat(self.img_files,self.depth_dir)
        assert len(self.img_files)==len(self.depth_files), f'image does not match depth'
###########################################################

    def main_code(self):
        #######################################################################
        sample_batch_idx = len(self.img_files)//self.batch_size
        fake_dic = []
        idx_simu=0
        for idx in range(sample_batch_idx):
            sample_air_batch_files = self.img_files[idx * self.batch_size:(idx + 1) * self.batch_size]
            sample_depth_batch_files = self.depth_files[idx * self.batch_size:(idx + 1) * self.batch_size]
            sample_air_batch = [np.array(Image.open(x).convert('RGB')).astype(np.float32) for x in sample_air_batch_files]
            sample_depth_batch = [np.array(sio.loadmat(x)['dph']).astype(np.float32)for x in sample_depth_batch_files]
            sample_air_images = np.array(sample_air_batch)
            sample_depth_images = np.expand_dims(sample_depth_batch, axis=3)
            fake_dic.append([sample_air_batch_files[0],0])
            illum=[[0.7,0.3,0],[0,1,0],[0,0,1]]
            i,j,task=0,0,0
            for un_illu in illum:
                j=0
                for water in range(0,10):
                    uw,B = self.wc_generator(sample_air_images,sample_depth_images,water_depth =np.random.uniform(0.4, 10,1), A=np.random.uniform(0.6, 0.9,1), water_type= water, water_illum=un_illu)
                    j=j+1
                    task= task+1
                    out_file = (self.img_files[idx].split(os.sep)[-1]).split('.')[0]
                    out_name = str(self.results_dir) + os.sep+f'{out_file}_{task}.png'
                    print(out_name)
                    idx_simu=idx_simu+1
                    fake_dic.append([out_name,task])
                    out_image = uw[0,:,:,:]
                    out_image = np.squeeze(out_image)
                    imageio.imsave(out_name, out_image)
                    out_name = str(self.results_dir) + os.sep+f'{out_file}_B{task}.png'
                    imageio.imsave(out_name, B[0,:,:,:])
                i = i + 1
        path1 = Path(str(Path(out_name).parent)+os.sep+'dic')
        if os.path.exists(path1):
            os.remove(path1)
        np.save(path1, fake_dic)
        path2 = Path(str(Path(out_name).parent)+os.sep+'dic.txt')
        if os.path.exists(path2):
            os.remove(path2)
        np.savetxt(path2, fake_dic, fmt='%s', newline='\n')
        path3 = Path(str(Path(out_name).parent)+os.sep+'dic.csv')
        if os.path.exists(path3):
            os.remove(path3)
        np_to_csv = pd.DataFrame(fake_dic)
        np_to_csv.to_csv(path3)

    def wc_generator(self,image,depth,A=0.7,water_type=1,water_depth=0,water_illum=[1,0,0]):
        """
        JM generator model
        :param z: input noise
        :param image: input clear image
        :param depth: input clear image depth map
        :return: downgraded image
        I(x) = J(x) exp(-eta*depth) + A(1-exp(-eta_haze*depth))
        """
        eta_r = np.array([0.30420412, 0.30474395, 0.35592191, 0.32493874, 0.55091001,0.42493874, 0.55874165, 0.13039252 , 0.10760831, 0.15963731])
        eta_g = np.array([0.11727661, 0.05999663, 0.11227639, 0.15305673, 0.14385827, 0.12305673, 0.0518615, 0.18667714, 0.1567016, 0.205724217])
        eta_b = np.array([0.1488851, 0.30099538, 0.38412464, 0.25060999, 0.01387215, 0.055799195, 0.0591001, 0.5539252 , 0.60103   , 0.733602  ])

        eta_rI,eta_gI,eta_bI=torch.tensor([[[eta_r[water_type]]]]), torch.tensor([[[eta_g[water_type]]]]), torch.tensor([[[eta_b[water_type]]]])
        eta = torch.stack([eta_rI, eta_gI, eta_bI],axis=3)
        print(f'attentuation: {eta}')
        depth =depth+water_depth
        eta =eta.detach().numpy()
        t = np.exp(np.multiply(-1, np.multiply(depth,eta)))
        #to array
        #I =L^kJ^ke^{-cd(x.y)} + (\kapa L_s^k/c^k)[1-e^{-cd(x,y)}]
        #l^k= w_aL_a+w_b L_b e^{-c(Z+d) +\sum_{i=1}^{N_c}w_{c,i}k_c P(x,y|L_c,\sigma)}e^{-cD} : Z: 摄像机和物体直接传播的额外距离， D:物体到额外光源的距离
        w_a, w_b, w_c = water_illum[0], water_illum[1], water_illum[2],
        L_a, L_b, L_c = 0.9,1,1
        Z_b= np.random.uniform(0,2)
        L_t1 = w_a*L_a
        L_t2 = w_b*L_b*np.exp(np.multiply(-1, np.multiply(depth+Z_b,eta)))
        x,y=np.meshgrid(np.linspace(0,depth.shape[2]-1,depth.shape[2]),np.linspace(0,depth.shape[1]-1,depth.shape[1]))
        sigma,z_l,r_l=np.random.uniform(0.2,1.1)*t.shape[2], np.random.uniform(-1,1),np.random.uniform(0.1,1)
        L_t3, v= self.get_art_light(x, y,eta, depth, sigma,x_c=np.random.randint(0,x.shape[1]),y_c=np.random.randint(0,x.shape[0]),L_art=0.75,Z_l= z_l,r_l=r_l)
        L_t=L_t1+L_t2+w_c*L_t3 #L_t3
#        L_tt=L_t1+L_t2+w_c*v
        direct =np.multiply( np.multiply(image,L_t),t )
#        direct1 =np.multiply( np.multiply(image,L_tt),t )
        eta_rI1,eta_gI1,eta_bI1=torch.tensor([[[eta_r[water_type]]]]), torch.tensor([[[eta_g[water_type]]]]), torch.tensor([[[eta_b[water_type]]]])
        eta_haze = torch.stack([eta_rI1,eta_gI1,eta_bI1],axis=3)
        eta_haze =eta_haze.detach().numpy()
        t_haze = np.exp(np.multiply(-1, np.multiply(depth,eta_haze)))
        image_haze = np.multiply( np.multiply(A*255, np.subtract(1.0, t_haze)), t)
        B = np.multiply( A*255, t)
        I = direct +  image_haze

        """
        if self.batch_size == 1:
            fig, ax = plt.subplots(1, 8, figsize=(3, 6))
            ax[0].imshow(np.squeeze(depth[0]))
            ax[1].imshow(np.squeeze(L_t2[0]))
            ax[2].imshow(np.squeeze(L_t3[0]))
            ax[3].imshow(np.squeeze(v[0]))
            ax[4].imshow(np.squeeze(image[0]) / 255)
            ax[5].imshow(np.squeeze(image_haze[0]) / 255)
            ax[6].imshow(np.squeeze(direct[0] + 0) / 255)
            ax[7].imshow(np.squeeze(I[0] + 0) / 255)
        else:
            fig, ax = plt.subplots(4, image.shape[0], figsize=(3, 3 * (image.shape[0])))
            [ax[0][i].imshow(np.squeeze(image_haze[i])/255) for i in range(image.shape[0])]
            [ax[1][i].imshow(np.squeeze(direct[i])/255) for i in range(image.shape[0])]
            [ax[2][i].imshow(np.squeeze(image[i])/255) for i in range(image.shape[0])]
            [ax[3][i].imshow(np.squeeze(I[i])/255) for i in range(image.shape[0])]
        """
        return I,B

    def get_art_light(self,x1, y1,eta,depth, sigma,x_c,y_c,L_art=0.8,r_l=0.3,Z_l=0.3):

        v = L_art * np.exp(-1.0 / (2 * sigma ** 2) * ( (x1-x_c) ** 2 + (y1-y_c) ** 2))
        v=np.expand_dims([v], axis=3)
        D= Z_l**2 +(0**2) * ((x1-x_c)**2+(y1-y_c)**2)
        D=np.expand_dims([np.sqrt(D)], axis=3)
        D_decay=np.exp(np.multiply(-1, np.multiply(D+depth,eta)))
        art_light = np.multiply(D_decay, v)
        return art_light,v


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=10, help='initial weights path')
    parser.add_argument('--batch-size', type=int, default=1, help='The size of image')
    parser.add_argument('--air-dataset',type=str, default=ROOT / 'air-image')
    parser.add_argument('--depth-dataset',type=str, default= ROOT /'air-depth')
    parser.add_argument('--results_dir',type=str, default= ROOT / 'uw-image')
    opt = parser.parse_known_args()[0]
    unwater = un_generator(batch_size=opt.batch_size,air_dir=opt.air_dataset,depth_dir=opt.depth_dataset,results_dir=opt.results_dir )
    unwater.main_code()






