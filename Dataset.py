# coding:UTF-8

import os
import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io as sio


def get_data_loaders(conf, status='train'):
    if status == 'test':
        test_dataset = TaskDataset(conf, 'test')
        test_loader = torch.utils.data.DataLoader(test_dataset)
        return test_loader
    if status == 'train':
        train_dataset = TaskDataset(conf, 'train')
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=conf.batch_size,
                                                   shuffle=True,
                                                   num_workers=conf.num_workers)
        return train_loader
    if status == 'val':
        val_dataset = TaskDataset(conf, 'val')
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=conf.batch_size,
                                                 num_workers=conf.num_workers)
        return val_loader


def read_path(path):
    imlist = []
    with open(path, 'r', encoding='utf-8') as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(impath)
    return imlist


class TaskDataset(Dataset):
    def __init__(self, conf, status='train'):
        # self.maskpath = conf.data_path + '/masks'
        if status == 'train':
            self.datapath = conf.data_path + '/train1'
            self.datalist = read_path('/home/Data/zhangyuhan1/five_fold_and_one_4_train_list.txt')
            # self.datalist = os.listdir(self.datapath)
            # print(self.datalist[0])
            # assert 0
            print("# train samples: {}".format(len(self.datalist)))
        if status == 'val':
            self.datapath = conf.data_path + '/train1'
            self.datalist = read_path('/home/Data/zhangyuhan1/five_fold_and_one_4_train_list.txt')
            # self.datalist = os.listdir(self.datapath)
            print("# val samples: {}".format(len(self.datalist)))
        if status == 'test':
            self.datapath = conf.data_path + '/train1'
            self.datalist = read_path('/home/Data/zhangyuhan1/five_fold_2_test_list.txt')
            # self.datalist = os.listdir(self.datapath)
            print("# test samples: {}".format(len(self.datalist)))

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):  # idx from [0,len(self)-1]
        filename = self.datalist[idx]
        # print('Processing: '+filename)
        # data1 = h5py.File(self.datapath + '\\' + filename)
        # data1 = sio.loadmat(self.datapath + '/' + filename)
        data1 = sio.loadmat(filename)

        # image1 = data1['image1'][:]
        # image1 = image1.reshape(1, image1.shape[0], image1.shape[1]).astype(np.float32)
        # image1 = np.transpose(image1, [0, 3, 1, 2])

        # image2 = data1['image2'][:]
        # image2 = image2.reshape(1, image2.shape[0], image2.shape[1]).astype(np.float32)
        # image2 = np.transpose(image2, [0, 3, 1, 2])

        images = data1['cube_local'][:]
        images = images.reshape(images.shape[0], images.shape[1], images.shape[2]).astype(np.float32)
        images = np.transpose(images, [2, 0, 1])
        images[images < 0] = 0
        images[images > 255] = 255

        labels = data1['label_local'][:]
        labels = labels.reshape(labels.shape[0], labels.shape[1], labels.shape[2]).astype(np.float32)
        labels = np.transpose(labels, [2, 0, 1])
        labels[labels < 0] = 0
        labels[labels > 255] = 255
        # labels = labels.reshape(1, labels.shape[0], labels.shape[1]).astype(np.float32)

        # istreat = data1['istreat'][:]
        # istreat = np.squeeze(istreat)

        # loops = data1['loops'][:]
        # loops = np.squeeze(loops)

        # data2 = h5py.File(self.maskpath + '\\' + filename)
        # data2 = sio.loadmat(self.maskpath + '/' + filename)
        # mask = data2['mask'][:]
        # mask = np.squeeze(mask)

        filename_sp = filename.split('/')
        imgname = filename_sp[-1]
        
        sample = {'images': torch.from_numpy(images/255),
                  # 'image2': torch.from_numpy(image2),
                  'labels': torch.from_numpy(labels/255),
                  # 'istreat': istreat,
                  'filename': imgname}

        return sample
