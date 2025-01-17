import time

from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import Dataset
import random
import cv2
import albumentations as A


def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1]).copy()


def chw_to_hwc(img):
    return np.transpose(img, axes=[1, 2, 0]).copy()


def read_img(filename, to_float=True):
    img = cv2.imread(filename)
    if to_float: img = img.astype('float32') / 255.0
    return img[:, :, ::-1]

def read_edgeimg(filename, to_float=True):
    img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
    if to_float: img = img.astype('float32') / 255.0
    return img[:, :]

def read_transimg(filename, to_float=True):
    img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
    if to_float: img = img.astype('float32') / 255.0
    return img[:, :]

def write_img(filename, img, to_uint=True):
    if to_uint: img = np.round(img * 255.0).astype('uint8')
    cv2.imwrite(filename, img[:, :, ::-1])



def augment(imgs=[], size=256, edge_decay=0., data_augment=True):
    H, W, _ = imgs[0].shape
    Hc, Wc = [size, size]

    # simple re-weight for the edge
    if random.random() < Hc / H * edge_decay:
        Hs = 0 if random.randint(0, 1) == 0 else H - Hc
    else:
        Hs = random.randint(0, H - Hc)

    if random.random() < Wc / W * edge_decay:
        Ws = 0 if random.randint(0, 1) == 0 else W - Wc
    else:
        Ws = random.randint(0, W - Wc)

    for i in range(len(imgs)):
        imgs[i] = imgs[i][Hs:(Hs + Hc), Ws:(Ws + Wc), :]

    if data_augment:
        # horizontal flip
        if random.randint(0, 1) == 1:
            for i in range(len(imgs)):
                imgs[i] = np.flip(imgs[i], axis=1)

        # bad data augmentations for outdoor dehazing
        rot_deg = random.randint(0, 3)
        for i in range(len(imgs)):
            imgs[i] = np.rot90(imgs[i], rot_deg, (0, 1))

    return imgs


def align(imgs=[], size=256):
    H, W, _ = imgs[0].shape
    Hc, Wc = [size, size]

    Hs = (H - Hc) // 2
    Ws = (W - Wc) // 2
    for i in range(len(imgs)):
        imgs[i] = imgs[i][Hs:(Hs + Hc), Ws:(Ws + Wc), :]

    return imgs


class PairLoader(Dataset):
    def __init__(self, root_dir, mode, degrade_type, size=256, edge_decay=0, data_augment=True, cache_memory=False):
        assert mode in ['train', 'valid', 'test']

        self.mode = mode
        self.size = size
        self.edge_decay = edge_decay
        self.data_augment = data_augment

        self.root_dir = root_dir
        self.degrade_type = degrade_type
        self.clean_img_names = sorted(os.listdir(os.path.join(self.root_dir, 'clean')))
        self.clean_img_num = len(self.clean_img_names)
        self.degra_img_names = sorted(os.listdir(os.path.join(self.root_dir, degrade_type)))
        self.degra_img_num = len(self.degra_img_names)

        self.cache_memory = cache_memory
        self.source_files = {}
        self.target_files = {}

    def __len__(self):
        return self.degra_img_num

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        # select a image pair
        if self.mode == 'train':
            clean_img_name = self.clean_img_names[idx]
            degra_img_name = self.degra_img_names[idx]

        if self.mode == 'test':
            #print("dataset test")
            clean_img_name = self.clean_img_names[idx]
            degra_img_name = self.degra_img_names[idx]

        if self.mode == 'valid':
            #print("dataset valid")
            clean_img_name = self.clean_img_names[idx]
            degra_img_name = self.degra_img_names[idx]

        # read images
        if degra_img_name not in self.source_files:
            source_img = read_img(os.path.join(self.root_dir, self.degrade_type, degra_img_name), to_float=False)
            target_img = read_img(os.path.join(self.root_dir, 'clean', clean_img_name), to_float=False)

            # cache in memory if specific (uint8 to save memory), need num_workers=0
            if self.cache_memory:
                self.source_files[degra_img_name] = source_img
                self.target_files[clean_img_name] = target_img
        else:
            # load cached images
            source_img = self.source_files[degra_img_name]
            target_img = self.target_files[clean_img_name]

        # [0, 1] to [-1, 1]
        source_img = source_img.astype('float32') / 255.0
        target_img = target_img.astype('float32') / 255.0
        #print(source_img.shape)

        # data augmentation
        if self.mode == 'train':
            [source_img, target_img] = augment([source_img, target_img], self.size, self.edge_decay, self.data_augment)

        if self.mode == 'valid':
            [source_img, target_img] = align([source_img, target_img], self.size)

        return {'source': hwc_to_chw(source_img), 'target': hwc_to_chw(target_img), 'filename': degra_img_name}


class Pair4typeLoader(Dataset):
    def __init__(self, root_dir, mode, degrade_type, size=256, edge_decay=0, data_augment=True, cache_memory=False):
        assert mode in ['train', 'valid', 'test']

        self.mode = mode
        self.size = size
        self.edge_decay = edge_decay
        self.data_augment = data_augment

        self.root_dir = root_dir
        self.degrade_type = degrade_type
        self.clean_img_names = sorted(os.listdir(os.path.join(self.root_dir, 'clean')))
        self.clean_img_num = len(self.clean_img_names)
        self.blur_img_names = sorted(os.listdir(os.path.join(self.root_dir, 'blur')))
        self.noise_img_names = sorted(os.listdir(os.path.join(self.root_dir, 'noise')))
        self.hazy_img_names = sorted(os.listdir(os.path.join(self.root_dir, 'haze')))
        self.dark_img_names = sorted(os.listdir(os.path.join(self.root_dir, 'dark')))
        self.hazy_img_num = len(self.hazy_img_names)

        self.cache_memory = cache_memory
        self.blur_files = {}
        self.noise_files = {}
        self.hazy_files = {}
        self.dark_files = {}

        self.target_files = {}

    def __len__(self):
        return self.hazy_img_num

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        # select a image pair
        if self.mode == 'train':
            clean_img_name = self.clean_img_names[idx]
            blur_img_name = self.blur_img_names[idx]
            noise_img_name = self.noise_img_names[idx]
            hazy_img_name = self.hazy_img_names[idx]
            dark_img_name = self.dark_img_names[idx]

        if self.mode == 'test':
            #print("dataset test")
            clean_img_name = self.clean_img_names[idx]
            blur_img_name = self.blur_img_names[idx]
            noise_img_name = self.noise_img_names[idx]
            hazy_img_name = self.hazy_img_names[idx]
            dark_img_name = self.dark_img_names[idx]

        if self.mode == 'valid':
            #print("dataset valid")
            clean_img_name = self.clean_img_names[idx]
            blur_img_name = self.blur_img_names[idx]
            noise_img_name = self.noise_img_names[idx]
            hazy_img_name = self.hazy_img_names[idx]
            dark_img_name = self.dark_img_names[idx]

        # read images
        if hazy_img_name not in self.hazy_files:
            blur_img = read_img(os.path.join(self.root_dir, 'blur', blur_img_name), to_float=False)
            noise_img = read_img(os.path.join(self.root_dir, 'noise', noise_img_name), to_float=False)
            hazy_img = read_img(os.path.join(self.root_dir, 'haze', hazy_img_name), to_float=False)
            dark_img = read_img(os.path.join(self.root_dir, 'dark', dark_img_name), to_float=False)
            target_img = read_img(os.path.join(self.root_dir, 'clean', clean_img_name), to_float=False)

            # cache in memory if specific (uint8 to save memory), need num_workers=0
            if self.cache_memory:
                self.blur_files[blur_img_name] = blur_img
                self.noise_files[noise_img_name] = noise_img
                self.hazy_files[hazy_img_name] = hazy_img
                self.dark_files[dark_img_name] = dark_img
                self.target_files[clean_img_name] = target_img
        else:
            # load cached images
            blur_img = self.blur_files[blur_img_name]
            noise_img = self.noise_files[noise_img_name]
            hazy_img = self.hazy_files[hazy_img_name]
            dark_img = self.dark_files[dark_img_name]
            target_img = self.target_files[clean_img_name]

        # [0, 1] to [-1, 1]
        blur_img = blur_img.astype('float32') / 255.0
        noise_img = noise_img.astype('float32') / 255.0
        hazy_img = hazy_img.astype('float32') / 255.0
        dark_img = dark_img.astype('float32') / 255.0
        target_img = target_img.astype('float32') / 255.0
        #print(source_img.shape)

        # data augmentation
        if self.mode == 'train':
            [blur_img, noise_img, hazy_img, dark_img, target_img] = augment([blur_img, noise_img, hazy_img, dark_img, target_img], self.size, self.edge_decay, self.data_augment)

        if self.mode == 'valid':
            [blur_img, noise_img, hazy_img, dark_img, target_img] = align([blur_img, noise_img, hazy_img, dark_img, target_img], self.size)

        return {'blur': hwc_to_chw(blur_img), 'noise': hwc_to_chw(noise_img), 'hazy': hwc_to_chw(hazy_img), 'dark': hwc_to_chw(dark_img), 'target': hwc_to_chw(target_img), 'filename': hazy_img_name}


class SingleLoader(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_names = sorted(os.listdir(self.root_dir))
        self.img_num = len(self.img_names)

    def __len__(self):
        return self.img_num

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        img_name = self.img_names[idx]
        img = read_img(os.path.join(self.root_dir, img_name), to_float= False)
        img = img.astype('float32') / 255.0

        return {'img': hwc_to_chw(img), 'filename': img_name}
