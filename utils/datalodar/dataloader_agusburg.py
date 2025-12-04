import scipy.io as sio
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
import random
from matplotlib import pyplot as plt
import h5py
def applyPCA(X, numComponents = 5):
    num_pixels =X.shape[0] * X.shape[1]
    num_bands = X.shape[2]
    data_2d = X.reshape(num_pixels, num_bands)
    data_2d_std = (data_2d - np.mean(data_2d, axis=0)) / np.std(data_2d, axis=0)
    pca = PCA(numComponents)
    pca.fit(data_2d_std)
    components = pca.components_
    projected = pca.transform(data_2d_std)
    projected_3d = projected.reshape(X.shape[0], X.shape[1], numComponents)
    return projected_3d

class LIDARHS(Dataset):
    def __init__(self, patchsize, mode = 'train', classnum = 100):
        if mode == 'train':
            print('train')

        self.mode = mode
        self.padding_layers = int(patchsize /2)
        self.lidar_mat = np.squeeze(sio.loadmat('/data_augsburg/LiDAR_data.mat')['LiDAR_data'].astype(np.float32))
        self.lidar_mat = np.pad(self.lidar_mat, self.padding_layers, mode='symmetric')
        self.sar_mat = np.squeeze(sio.loadmat('/data_augsburg/SAR_data.mat')['SAR_data'].astype(np.float32))
        self.sar_mat = np.pad(self.sar_mat, ((self.padding_layers, self.padding_layers), (self.padding_layers, self.padding_layers), (0, 0)),
                        mode='symmetric')
        self.HS_mat = np.squeeze(sio.loadmat('/data_augsburg/HSI_data.mat')['HSI_data'].astype(np.float32))

        self.HS_mat = applyPCA(self.HS_mat, 30)

        self.HS_mat = np.pad(self.HS_mat, ((self.padding_layers, self.padding_layers), (self.padding_layers, self.padding_layers), (0, 0)),
                        mode='symmetric')

        self.train_lable_os = '/data_augsburg/TrainImage.mat'
        self.test_lable_os = '/data_augsburg/All_Label.mat'
        self.lable_mat = np.squeeze(sio.loadmat(self.test_lable_os)['All_Label'].astype(np.float32))

        self.lidar = []
        self.HS = []
        self.sar = []
        self.lable = []
        self.ik = []
        for i in range(0, len(self.lable_mat)):
            for k in range(0, len(self.lable_mat[0])):
                ik_now = (i,k)
                lable_now = self.lable_mat[i][k]
                if lable_now != 0:
                    self.lable.append(lable_now)
                    lidar_now = self.lidar_mat[i:i + (self.padding_layers*2+1), k:k + (self.padding_layers*2+1)]
                    # lidar_now = self.lidar_mat[i:i + (self.padding_layers * 2), k:k + (self.padding_layers * 2)]

                    lidar_now = np.expand_dims(lidar_now, axis=0)
                    HS_now = self.HS_mat[i:i + (self.padding_layers*2+1), k:k + (self.padding_layers*2+1)]
                    # HS_now = self.HS_mat[i:i + (self.padding_layers*2), k:k + (self.padding_layers*2)]

                    HS_now = np.transpose(HS_now, (2, 0, 1))
                    sar_now = self.sar_mat[i:i + (self.padding_layers*2+1), k:k + (self.padding_layers*2+1)]
                    # sar_now = self.sar_mat[i:i + (self.padding_layers*2), k:k + (self.padding_layers*2)]
                    sar_now = np.transpose(sar_now, (2, 0, 1))
                    self.sar.append(sar_now)
                    self.lidar.append(lidar_now)
                    self.HS.append(HS_now)
                    self.ik.append(ik_now)

        # print(np.array(self.lidar).shape,np.array(self.HS).shape,np.array(self.lable).shape)
        # (2832, 1, 17, 17)(2832, 30, 17, 17)(2832, )
        if mode == 'train':
        #
            self.train_hsi = []
            self.train_lidar = []
            self.train_sar = []
            self.train_lable = []
            self.train_ik = []
            for cls in range(1,int(max(self.lable)) + 1):
                indices = [index for index, value in enumerate(self.lable) if value == cls]
                random_indices = random.sample(indices, classnum)
                self.train_hsi += [self.HS[index] for index in random_indices]
                self.train_lidar += [self.lidar[index] for index in random_indices]
                self.train_sar += [self.sar[index] for index in random_indices]
                self.train_lable += [self.lable[index] for index in random_indices]
                self.train_ik += [self.ik[index] for index in random_indices]
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_lable)
        else:
            return len(self.lable)
    def __getitem__(self, item):
        if self.mode == 'train':
            lidar, HS, sar, lable, ik = self.train_lidar[item], self.train_hsi[item], self.train_sar[item],self.train_lable[item], self.train_ik[item]
        else:
            lidar, HS, sar, lable, ik = self.lidar[item], self.HS[item],self.sar[item], self.lable[item], self.ik[item]
        # lidar, HS, sar, lable, ik = self.lidar[item], self.HS[item],self.sar[item], self.lable[item], self.ik[item]
        self.dataset = 1
        if self.dataset == 0:return lidar, HS, lable, ik
        if self.dataset == 1:return sar, HS, lable, ik
        if self.dataset == 2:return sar, HS, lidar, lable, ik


if __name__ == '__main__':

    data = LIDARHS(32, mode='train')
    train_loader = DataLoader(data, batch_size = 4, shuffle = True, num_workers = 1)
    device = 'cpu'

    for step, (lidar_out, HS_out, lable_out, ik_out) in enumerate(train_loader):
        lidar_out = lidar_out.type(torch.float).to(device)
        HS_out = HS_out.type(torch.float).to(device)
        lable_out = lable_out.type(torch.LongTensor).to(device) - 1
        # i = ik_out[:][0]
        # k = ik_out[:][1]
        # if step == 1:
        #     print(lable_out[0])
        #     print(ik_out)
        #     print(i[0])
        #     print(k[0])
        #     print(lidar_out[0])
        # 8*8patch，[4,4]center

        # * * * * * * * *
        # * * * * * * * *
        # * * * * * * * *
        # * * * * * * * *
        # * * * * c * * *
        # * * * * * * * *
        # * * * * * * * *
        # * * * * * * * *

        if step == 1:
            from matplotlib import pyplot as plt
            print(lable_out,type(lable_out),lable_out.shape)
            print(lidar_out.shape)
            print(HS_out.shape)
            # print(HS_out[0][1][ :, :])
            # print(lable_out[0])
            plt.subplot(1, 2, 1)
            plt.imshow(HS_out[0][1][ :, :], cmap='gray')
            plt.subplot(1, 2, 2)
            plt.imshow(lidar_out[0][0], cmap='gray')
            plt.show()
