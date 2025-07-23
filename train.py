
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

class Opion():
    
    def __init__(self):
        self.dataroot= r'첫 번째 스테이지(First stage)의 정답 2D PCB 위상 맵(Phase map) 폴더 경로'
        self.maskroot= r'그림자와 빛 반사 영역에 대한 마스크 영상 폴더 경로'
        self.unwraproot = r'두 번재 스테이지(Second stage)의 정답 2D PCB 펼쳐진 위상 맵(Unwrap) 폴더 경로'
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

# 학습 데이터 전처리
opt = Opion()
transform_mask = transforms.Compose([transforms.ToTensor()]) # 마스크 데이터 설정
transform = transforms.Compose([ transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]) # 영상 데이터 설정
dataset_train = Data_load(opt.dataroot, opt.maskroot,opt.unwraproot,transform, transform_mask) # 데이터 불러오기
iterator_train = (data.DataLoader(dataset_train, batch_size=opt.batchSize,shuffle=False))

# 모델 생성, 불러오기, 저장
model = create_model(opt)
load_epoch=<숫자>
model.load("저장된 모델 경로/" + str(load_epoch))
save_dir = "모델 저장할 경로"

# 학습 수, 학습 시간, 로그 TXT 메모장 생성 (loss 확인)
total_steps = 0
iter_no = 0
iter_start_time = time.time()
logger = Logger('/loss.txt', title='MIN')
logger.set_names(['Grad_loss', 'Entropy_loss', 'Std_dev_loss', 'G_GAN', 'loss_G_L1', 'loss_D_fake', 'loss_F_fake'])

# 학습 시작
for epoch in range(501):
    iter_no += 1
    epoch_start_time = time.time()
    epoch_iter = 0

    # 10번 학습마다 모델 저장
    if epoch % 10 == 0:
            model.save(save_dir+'/model/' + str(epoch))
    
    # 데이터셋 속 첫 번째 스테이지 영상, 마스크, 두 번째 스테이지 영상을 이용하여 학습 진행
    for image, mask, unwrap,index in (iterator_train):
        image=image.cuda()
        unwrap = unwrap.cuda()
        mask=mask.cuda()
        mask=mask[0][0]
        mask=torch.unsqueeze(mask,0)
        mask=torch.unsqueeze(mask,1)
        mask=mask.byte()
                
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        
        # 전처리된 데이터를 모델에 입력
        model.set_input(image,mask,unwrap,index)
        model.set_gt_latent()

        # 모델 loss 갱신
        loss_Grad , loss_Entropy , loss_Std_dev, loss_G_GAN , loss_G_L1, loss_D_fake, loss_F_fake = model.optimize_parameters()
        
        # display_freq 수만큼 반복문 후 결과 영상 저장
        if total_steps %opt.display_freq== 0:
            real_A,real_B,fake_P, real_C, real_D, fake_B=model.get_current_visuals()
        
        # 모델 학습 시간, 오차 계산
        if total_steps %1== 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            print(errors)
    
    # 10번 학습마다 로그 TXT에 loss 저장
    if epoch % 10 == 0 and epoch != 0:
        logger.append([loss_Grad , loss_Entropy , loss_Std_dev,loss_G_GAN ,  loss_G_L1, loss_D_fake, loss_F_fake])
    
    # 매 학습마다 학습 수, 학습 시간, 학습에 대한 정보 출력
    print('End of epoch %d / %d \t Time Taken: %d sec' %
            (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    
    # 학습률 갱신
    model.update_learning_rate()
# 로그 TXT 닫기
logger.close()
