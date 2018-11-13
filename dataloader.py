import torch
import cv2
import os
import pickle
import pandas as pd
import random
from PIL import Image,ImageFilter,ImageEnhance
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import config as cfg


img_ZJL = pickle.load(open('./pkl/img_ZJL.pkl', 'rb'))
catogery_word = pickle.load(open('./pkl/catogery_word.pkl', 'rb'))
ZJL_catogery = pickle.load(open('./pkl/ZJL_catogery.pkl', 'rb'))
ZJL_label = pd.read_csv(os.path.join(cfg.SPLIT_PATH, 'LabelEnc.csv'), index_col=0, names=['Class','Index'])['Index']
ZJL_label_test = pickle.load(open('./pkl/ZJL_label_test.pkl', 'rb'))


def color_aug(image):
    # 亮度增强
    if random.choice([0, 1]):
        enh_bri = ImageEnhance.Brightness(image)
        brightness = random.choice([0.5, 0.8, 1.2, 1.5])
        image = enh_bri.enhance(brightness)
    # 色度增强
    if random.choice([0, 1]):
        enh_col = ImageEnhance.Color(image)
        color = random.choice([0.5, 0.8, 1.2, 1.5])
        image = enh_col.enhance(color)
    # 对比度增强
    if random.choice([0, 1]):
        enh_con = ImageEnhance.Contrast(image)
        contrast = random.choice([0.5, 0.8, 1.2, 1.5])
        image = enh_con.enhance(contrast)
    # 锐度增强
    if random.choice([0, 1]):
        enh_sha = ImageEnhance.Sharpness(image)
        sharpness = random.choice([0.5, 0.8, 1.2, 1.5])
        image = enh_sha.enhance(sharpness)
    # mo hu
    if random.choice([0, 1]):
        image = image.filter(ImageFilter.BLUR)
    return image

def Random_Rotation(PIL_img):
    possible = np.random.randint(2)
    if possible == 0:
        return PIL_img

    degree = np.random.randint(-45, 46)  #45-30
    img_rot = PIL_img.rotate(degree)

    return img_rot

def Random_flip(PIL_img):
    possible = np.random.randint(2)
    if possible == 0:
        return PIL_img
    img_fp = PIL_img.transpose(Image.FLIP_LEFT_RIGHT)
    return img_fp


def Random_Size(PIL_img):
    possible = np.random.randint(2)
    if possible == 0:
        return PIL_img
    w, h = PIL_img.size
    r = np.random.randint(4)/10
    img_sz = PIL_img.crop((- w * r, - h * r, w * (1 + r), h * (1 + r))).resize((64, 64))

    return img_sz

def Random_Move(PIL_img):
    possible = np.random.randint(2)
    if possible == 0:
        return PIL_img
    w, h = PIL_img.size
    rx = np.random.randint(-2, 3)/10
    ry = np.random.randint(-2, 3) / 10
    img_mv = PIL_img.crop((- w * rx, - h * ry, w*(1-rx), h*(1-ry)))
    return img_mv


def img_normalizer(PIL_img):

    normalize = transforms.Normalize(
        # mean=[0.485, 0.456, 0.406],
        mean=[0.51052445, 0.45063994, 0.41973213],
        # std=[0.229, 0.224, 0.225]
        std=[0.26807685, 0.25218709, 0.2435979]
    )

    preprocess_img = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    img = preprocess_img(PIL_img)
    if img.size()[0] == 1 :
        img = torch.cat((img, img, img), 0)
    return img

def data_aug(PIL_img):
    #PIL_img = color_aug(PIL_img)
    img_rot = Random_Rotation(PIL_img)
    img_flip = Random_flip(img_rot)
    img_size = Random_Size(img_flip)
    img_mv = Random_Move(img_size)

    return img_flip



class Trainset(Dataset):
    def __init__(self, img_list, normalizer=img_normalizer):

        self.img_list = img_list
        self.normalizer = normalizer
        self.data_aug= data_aug

    def __getitem__(self, index):

        img_name = self.img_list[index]
        ZJL = img_ZJL[img_name]
        catogery = ZJL_catogery[ZJL]
        label = ZJL_label[ZJL]

        img = os.path.join(cfg.IMG_PATH, img_name)
        PIL_img = Image.open(img)
        PIL_img = PIL_img.resize((cfg.IMPUT_SIZE, cfg.IMPUT_SIZE))
        img = self.data_aug(PIL_img)
        img_tensor = self.normalizer(img)

        label = torch.tensor(int(label))

        return img_tensor, label

    def __len__(self):
        return len(self.img_list)


class Testset(Dataset):
    def __init__(self, img_list, normalizer=img_normalizer):

        self.img_list = img_list
        self.normalizer = normalizer
        self.data_aug= data_aug

    def __getitem__(self, index):

        img_name = self.img_list[index]
        ZJL = img_ZJL[img_name]
        catogery = ZJL_catogery[ZJL]
        label = ZJL_label_test[ZJL]

        img = os.path.join(cfg.IMG_PATH, img_name)
        PIL_img = Image.open(img)
        #PIL_img = PIL_img.resize( (256,256 ) )
        img = self.data_aug(PIL_img)
        img_tensor = self.normalizer(img)

        label = torch.tensor(int(label))

        return img_tensor, label

    def __len__(self):
        return len(self.img_list)