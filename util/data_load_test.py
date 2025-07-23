import random
import torch
from PIL import Image
from glob import glob
import os
import json
import pandas as pd
import torchvision
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
import PIL
from numba import jit
import pylab as plt
import numpy as np
import datatable as dt
from skimage.restoration import unwrap_phase

class Data_load(torch.utils.data.Dataset):
    def __init__(self, img_root,mask_root, img_transform, mask_transform):
        super(Data_load, self).__init__()
        self.img_transform = img_transform # 영상 전처리
        self.mask_transform = mask_transform # 마스크 전처리
        # 데이터 폴더 속 모든 지정 형식(bmp) 파일 로드
        self.paths = sorted(glob('{:s}/*.bmp'.format(img_root)))
        self.mask_paths = sorted(glob('{:s}/*.bmp'.format(mask_root)))
      
    def __getitem__(self, index):
        # 영상 및 마스크 크기 재설정
        gt_image = cv2.imread(self.paths[index])
        mask = cv2.imread(self.mask_paths[index])
        gt_image = cv2.resize(gt_image,(256,256))
        mask = cv2.resize(mask,(256,256))
        
        # 영상 및 마스크 전처리
        gt_img = Image.fromarray(gt_image)
        gt_img = self.img_transform(gt_img.convert('RGB'))
        maskk = Image.fromarray(mask)
        maskk = self.mask_transform(maskk.convert('RGB'))
        
        return gt_img , maskk, index

    def __len__(self):
        return len(self.paths)
        
