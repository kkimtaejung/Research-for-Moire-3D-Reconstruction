import warnings
warnings.filterwarnings(action='ignore')
import time
from util.data_load import Data_load
from models.models import create_model
import torch
import os
import torchvision
from torch.utils import data
import torchvision.transforms as transforms
import cv2
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numba import jit
from skimage.restoration import unwrap_phase
from utils import Logger, AverageMeter, savefig
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize
from keras.datasets import cifar10
import pylab as plt
from collections import OrderedDict
from torch.autograd import Variable
import torch.nn.functional as F
import datatable as dt
import pandas as pd
from numba import types, typed
import PIL

class Opion():
    
    def __init__(self):
        self.dataroot= r'그림자와 빛 반사가 존재하는 2D PCB 위상 맵(Phase map) 폴더 경로'
        self.maskroot= r'그림자와 빛 반사 영역에 대한 마스크 영상 폴더 경로'
        self.batchSize= 1  # 배치 사이즈
        self.fineSize=256 # 이미지 사이즈
        self.input_nc=3  # 첫 번째 스테이지 입력 채널
        self.input_nc_g=6 # 두 번째 스테이지 입력 채널
        self.output_nc=3# 최종 출력 채널
        self.ngf=64 # inner 채널
        self.which_model_netD='basic' # 판별자: patch discriminator
        
        self.which_model_netF='feature'# 판별자: feature patch discriminator
        self.which_model_netG='unet_256'# 첫 번째 스테이지
        self.which_model_netP='unet_256'# 두 번째 스테이지
        self.triple_weight=1
        self.name='Moire Inpainting Network'
        self.n_layers_D='3' # 네트워크 깊이
        self.gpu_ids=[0] # GPU 번호
        self.model='MIN_net' # 본격적인 모델 선택 파트
        self.checkpoints_dir=r'.\checkpoints' # 이어서 학습
        self.norm='instance' # 정규화 과정
        self.fixed_mask=1 # 마스크 값 설정
        self.use_dropout=False # dropout 설정
        self.init_type='normal' 
        self.lambda_A=100
        self.threshold=5/16.0
        self.stride=1
        self.shift_sz=1
        self.mask_thred=1
        self.bottleneck=512
        self.gp_lambda=10.0
        self.ncritic=5
        self.constrain='Adam' # 옵티마이저
        self.strength=1
        self.init_gain=0.02
        self.cosis=1
        self.gan_type='lsgan'
        self.gan_weight=0.2
        self.overlap=4
        self.skip=0
        self.display_freq=10
        self.print_freq=50
        self.save_latest_freq=5000
        self.save_epoch_freq=5
        self.continue_train=False
        self.epoch_count=1
        self.phase='train'
        self.which_epoch=''
        self.niter=20
        self.niter_decay=10000
        self.beta1=0.5
        self.lr=0.0002 # 학습률
        self.lr_policy='lambda'
        self.lr_decay_iters=50
        self.isTrain=True

# 테스트 데이터 전처리
opt = Opion()
transform_mask = transforms.Compose([transforms.ToTensor()]) # 마스크 데이터 설정
transform = transforms.Compose([ transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]) # 영상 데이터 설정
dataset_train = Data_load(opt.dataroot, opt.maskroot,opt.unwraproot,transform, transform_mask) # 데이터 불러오기
iterator_train = (data.DataLoader(dataset_train, batch_size=opt.batchSize,shuffle=False))

# 학습된 모델 불러오기
model = create_model(opt)
load_epoch=<숫자>
model.load("저장된 모델 경로/" + str(load_epoch))

# 테스트 시작
for image, mask, index in (iterator_test):
    
    # 테스트 데이터 전처리
    iter_start_time = time.time()
    image=image.cuda()
    mask=mask.cuda()
    mask=mask[0][0]
    mask=torch.unsqueeze(mask,0)
    mask=torch.unsqueeze(mask,1)
    mask=mask.byte()
    
    # 전처리된 데이터 모델에 입력, 테스트 결과 시각화
    model.set_input(image,mask)
    model.set_gt_latent()
    model.test()
    real_A,real_B,fake_B=model.get_current_visuals()
   
    iter_end_time = time.time()
    inference_time = iter_end_time - iter_start_time
    print(f"Inference time for iteration {i}: {inference_time:.4f} seconds")

