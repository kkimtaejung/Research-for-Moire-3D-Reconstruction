#-*-coding:utf-8-*-
import pylab as plt
import torch
from collections import OrderedDict
from torch.autograd import Variable
from numba import jit
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks
from .vgg16 import vgg19
import datatable as dt
import pandas as pd
import numpy as np
from numba import types, typed
import matplotlib.pyplot as plt
import PIL
import cv2
from .pid import PIDAdam
import numpy as np
import os
import sys
from .mvtec_ad.model import Generator, Encoder
import copy
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from skimage.restoration import unwrap_phase

# 복원된 펼쳐진 위상 맵의 균일함을 평가하기 위한 손실함수
# 기울기 크기 (에지 변화)
# 엔트로피 평균 (픽셀 값의 복잡성)
# 표준편차 평균 (전체 픽셀 분포 폭)
def Uniform_loss(tensor):
    # tensor의 shape은 [batch, channel, height, width]
    batch_size, _, height, width = tensor.shape
    # 배치 내의 각 이미지에 대해 기울기의 평균 크기를 저장할 리스트
    gradient_magnitudes_means = []
    entropy_means = []
    std_deviation_means = []
    # 배치 내의 각 이미지에 대한 루프
    for i in range(batch_size):
        # 현재 이미지 선택 (채널이 여러 개인 경우 평균을 사용하여 하나의 흑백 이미지로 변환)
        image_data = tensor[i].mean(dim=0)
        # 기울기 계산을 위해 이미지 데이터를 numpy 배열로 변환
        image_data_np = image_data.cpu().numpy()
        # 기울기 계산
        gradient = np.gradient(image_data_np)
        gradient_magnitude = np.sqrt(gradient[0]**2 + gradient[1]**2)
        # 엔트로피 계산을 위한 히스토그램 계산
        histogram, _ = np.histogram(image_data_np, bins=np.arange(0, 256), density=True)
        histogram = histogram[histogram > 0]  # 확률이 0인 값을 제거
        entropy = -np.sum(histogram * np.log2(histogram))
        # 표준 편차 계산
        std_deviation = np.std(image_data_np)
        # 현재 이미지에 대한 기울기 크기의 평균, 엔트로피, 표준 편차를 리스트에 추가
        gradient_magnitudes_means.append(np.mean(gradient_magnitude))
        entropy_means.append(entropy)
        std_deviation_means.append(std_deviation)
    # 배치 내의 모든 이미지에 대한 기울기 크기의 평균, 엔트로피의 평균, 표준 편차의 평균을 계산 및 반환
    grad_mean = np.mean(gradient_magnitudes_means)
    entropy_mean = np.mean(entropy_means)
    std_dev_mean = np.mean(std_deviation_means)
    # 세 가지 측정값의 평균을 반환
    return grad_mean , entropy_mean , std_deviation_means

# 복원된 펼쳐진 위상 맵의 높이 차이가 균일하다는 점을 고려하여 설계된 손실함수
def height_difference_loss(tensor):
    # tensor의 크기는 [batch, channel, height, width] 라고 가정
    above = torch.abs(tensor[:, :-2])  # [w,h-1,c]
    center = torch.abs(tensor[:, 1:-1])  # [w,h,c]
    below = torch.abs(tensor[:, 2:])  # [w,h+1,c]
    # [w,h-1,c]과 [w,h+1,c]이 0이 아닌 위치의 마스크 생성
    mask_above = (above != 0)
    mask_below = (below != 0)
    # 두 마스크 모두 True인 위치에서만 loss 적용
    mask = mask_above & mask_below
    loss = torch.abs(above - center) - torch.abs(center - below)
    masked_loss = torch.abs(loss) * mask.float()
    # 마스크된 위치의 평균 loss 반환
    return torch.sum(masked_loss) / torch.sum(mask)

# Moire Inpainting Network(MIN) 모델
class MIN(BaseModel):
    def name(self):
        return 'MINModel'
   
    # 모델 초기화
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.device = torch.device('cuda')
        self.opt = opt
        self.isTrain = opt.isTrain
        self.vgg=vgg19(requires_grad=False)
        self.vgg=self.vgg.cuda()
        self.vgg2=vgg19(requires_grad=False)
        self.vgg2=self.vgg2.cuda()
        
        # Phase_noise: 그림자와 빛 반사가 존재하는 PCB 위상 맵(Phase map)
        # Phase_real: 정답 PCB 위상 맵(Real Phase map)
        # Unwrap_noise: 그림자와 빛 반사가 존재하는 PCB 펼쳐진 위상 맵(2D Unwrap)
        # Unwrap_real: 정답 PCB 펼쳐진 위상 맵(Real 2D Unwrap)
        self.Phase_noise = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.Unwrap_noise = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.Phase_real = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize)
        self.Unwrap_real = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize)

        # mask_Phase: 위상 맵에서의 마스크 영역 정의
        # mask_Unwrap: 펼쳐진 위상 맵에서의 마스크 영역 정의
        self.mask_Phase = torch.ByteTensor(1, 1, opt.fineSize, opt.fineSize)
        self.mask_Phase.zero_()
        self.mask_Phase[:, :, int(self.opt.fineSize/4) + self.opt.overlap : int(self.opt.fineSize/2) + int(self.opt.fineSize/4) - self.opt.overlap,\
                                int(self.opt.fineSize/4) + self.opt.overlap: int(self.opt.fineSize/2) + int(self.opt.fineSize/4) - self.opt.overlap] = 1
        self.mask_Unwrap = torch.ByteTensor(1, 1, opt.fineSize, opt.fineSize)
        self.mask_Unwrap.zero_()
        self.mask_Unwrap[:, :, int(self.opt.fineSize/4) + self.opt.overlap : int(self.opt.fineSize/2) + int(self.opt.fineSize/4) - self.opt.overlap,\
                                int(self.opt.fineSize/4) + self.opt.overlap: int(self.opt.fineSize/2) + int(self.opt.fineSize/4) - self.opt.overlap] = 1
        self.mask_type = opt.mask_type
        self.gMask_opts = {}
        if len(opt.gpu_ids) > 0:
            self.use_gpu = True
            self.mask_Phase = self.mask_Phase.cuda()
            self.mask_Unwrap = self.mask_Unwrap.cuda()
        self.netG,self.Cosis_list,self.Cosis_list2, self.MIN_model= networks.define_G(opt.input_nc_g, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt, self.mask_Unwrap, opt.norm, opt.use_dropout, opt.init_type, self.gpu_ids, opt.init_gain)
        
        # netP: 첫 번째 스테이지 생성자
        # netP2: 두 번째 스테이지 생성자
        self.netP,_,_,_=networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                    opt.which_model_netP, opt, self.mask_Unwrap, opt.norm, opt.use_dropout, opt.init_type, self.gpu_ids, opt.init_gain)
        self.netP2,_,_,_=networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                    opt.which_model_netP, opt, self.mask_Unwrap, opt.norm, opt.use_dropout, opt.init_type, self.gpu_ids, opt.init_gain)
        self.Cosis_list2 = copy.copy(self.Cosis_list)
        self.Cosis_list22 = copy.copy(self.Cosis_list2)
        if self.isTrain:
            use_sigmoid = False
            if opt.gan_type == 'vanilla':
                use_sigmoid = True  
            # 첫 번째 스테이지 판별자 D,F
            self.netD = networks.define_D(opt.input_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids, opt.init_gain)
            self.netF = networks.define_D(opt.input_nc, opt.ndf,
                                          opt.which_model_netF,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                          opt.init_gain) 
            # 두 번째 스테이지 판별자 D2,F2
            self.netD2 = networks.define_D(opt.input_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids, opt.init_gain)
            self.netF2 = networks.define_D(opt.input_nc, opt.ndf,
                                          opt.which_model_netF,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                          opt.init_gain)            
        if not self.isTrain or opt.continue_train:
            print('Loading pre-trained network!')
            self.load_network(self.netP, 'P', opt.which_epoch)
            self.load_network(self.netP2, 'P2', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)
                self.load_network(self.netF, 'F', opt.which_epoch)
                self.load_network(self.netD2, 'D2', opt.which_epoch)
                self.load_network(self.netF2, 'F2', opt.which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            
            # 손실함수 스테이지별로 각각 정의
            self.criterionGAN = networks.GANLoss(gan_type=opt.gan_type, tensor=self.Tensor)
            self.criterionGAN2 = networks.GANLoss(gan_type=opt.gan_type, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL12 = torch.nn.L1Loss()

            # 각각의 판별자, 생성자에 따른 옵티마이저 할당
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_P = torch.optim.Adam(self.netP.parameters(),
                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_P2 = torch.optim.Adam(self.netP2.parameters(),
                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_F = torch.optim.Adam(self.netF.parameters(),
                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D2 = torch.optim.Adam(self.netD2.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_F2 = torch.optim.Adam(self.netF2.parameters(),
                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_P2)
            self.optimizers.append(self.optimizer_P)
            
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_F)
            self.optimizers.append(self.optimizer_D2)
            self.optimizers.append(self.optimizer_F2)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

            # 각각의 판별자, 생성자 네트워크 초기화
            print('---------- Networks initialized -------------')
            networks.print_network(self.netP2)
            networks.print_network(self.netP)
            
            if self.isTrain:
                networks.print_network(self.netD)
                networks.print_network(self.netF)
                networks.print_network(self.netD2)
                networks.print_network(self.netF2)
            print('-----------------------------------------------')

    # 모델에 입력되는 데이터 사전처리
    def set_input(self,input,mask,unwrap, index):
        self.index = index
        Phase_noise = input
        Phase_real = input.clone()
        Unwrap_noise = unwrap
        Unwrap_real = unwrap.clone()
        input_mask=mask

        self.Phase_noise.resize_(Phase_noise.size()).copy_(Phase_noise)
        self.Phase_real.resize_(Phase_real.size()).copy_(Phase_real)
        self.Unwrap_noise.resize_(Unwrap_noise.size()).copy_(Unwrap_noise)
        self.Unwrap_real.resize_(Unwrap_real.size()).copy_(Unwrap_real)

        self.image_paths = 0

        # 각 스테이지 별 마스크 영역을 영상에 덮어 생성
        if self.opt.mask_type == 'center':
            self.mask_Phase=self.mask_Phase
            self.mask_Unwrap=self.mask_Unwrap

        self.ex_mask = self.mask_Phase.expand(1, 3, self.mask_Phase.size(2), self.mask_Phase.size(3)) # 1*c*h*w
        self.ex_mask2 = self.mask_Unwrap.expand(1, 3, self.mask_Unwrap.size(2), self.mask_Unwrap.size(3)) # 1*c*h*w
        self.inv_ex_mask = torch.add(torch.neg(self.ex_mask.float()), 1).byte()
        self.mask_Phase = self.mask_Phase.bool()
        self.inv_ex_mask2 = torch.add(torch.neg(self.ex_mask2.float()), 1).byte()
        self.mask_Unwrap = self.mask_Unwrap.bool()
        self.Phase_noise.narrow(1,0,1).masked_fill_(self.mask_Phase, 1)
        self.Phase_noise.narrow(1,1,1).masked_fill_(self.mask_Phase, 1)
        self.Phase_noise.narrow(1,2,1).masked_fill_(self.mask_Phase, 1)
        self.set_latent_mask(self.mask_Unwrap, 3, self.opt.threshold)

    def set_latent_mask(self, mask_Unwrap, layer_to_last, threshold):
        self.MIN_model[0].set_mask(mask_Unwrap, layer_to_last, threshold)
        self.Cosis_list[0].set_mask(mask_Unwrap, self.opt)
        self.Cosis_list2[0].set_mask(mask_Unwrap, self.opt)
        self.Cosis_list2[0].set_mask(mask_Unwrap, self.opt)
        self.Cosis_list22[0].set_mask(mask_Unwrap, self.opt)

    # f-AnoGAN 학습된 모델을 활용하여 이상치 영상(Anomal image) 취득
    def anomal_detect(self,test_root,nnum, n_grid_lines=1, force_download=False, latent_dim=100, img_size=256, channels=1, n_iters=None):
        class Opt:
            def __init__(self, test_root, n_grid_lines, force_download, latent_dim, img_size, channels, n_iters):
                self.test_root = test_root
                self.n_grid_lines = n_grid_lines
                self.force_download = force_download
                self.latent_dim = latent_dim
                self.img_size = img_size
                self.channels = channels
                self.n_iters = n_iters

        opt = Opt(test_root, n_grid_lines, force_download, latent_dim, img_size, channels, n_iters)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        pipeline = [transforms.Resize([opt.img_size]*2)]
        if opt.channels == 1:
            pipeline.append(transforms.Grayscale())
        pipeline.extend([transforms.ToTensor(),
                        transforms.Normalize([0.5]*opt.channels, [0.5]*opt.channels)])

        transform = transforms.Compose(pipeline)
        dataset = ImageFolder(opt.test_root, transform=transform)
        dataloader = DataLoader(dataset, batch_size=n_grid_lines, shuffle=False)

        sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
        
        generator = Generator(opt)
        encoder = Encoder(opt)
        if nnum == 1:
            generator.load_state_dict(torch.load("models/results/PCB1/generator"))
            encoder.load_state_dict(torch.load("models/results/PCB1/encoder"))
        elif nnum ==2:
            generator.load_state_dict(torch.load("models/results/PCB2/generator"))
            encoder.load_state_dict(torch.load("models/results/PCB2/encoder"))
        else:
            generator.load_state_dict(torch.load("models/results/PCB3/generator"))
            encoder.load_state_dict(torch.load("models/results/PCB3/encoder"))
        
        generator.to(device).eval()
        encoder.to(device).eval()

        for i, (img, _) in enumerate(dataloader):  
            real_img = img.to(device)

            real_z = encoder(real_img)
            fake_img = generator(real_z)

            anomal_image = real_img - fake_img

        return anomal_image
    
    # PAM(Position Adaptive Module): 이상치 영상을 활용하여 마스크 영상 생성
    def anomal_mask(self,anomal_image):
        mask = np.zeros((256,256))
        a = np.zeros((256))
        for i in range(256):
            for j in range(256):
                a[int(anomal_image[i][j])] += 1
        max_value = np.max(a)
        max_index = np.unravel_index(np.argmax(a), a.shape)

        for i in range(256):
            for j in range(256):
						
                if anomal_image[i][j] > int(max_index[0])+18 or anomal_image[i][j] < int(max_index[0])-18:
                    mask[i][j] = 255
                else:
                    mask[i][j] = 0
                
        return mask

    # 모델 학습 과정
    def forward(self):
        # real_A: 그림자와 빛 반사가 존재하는 PCB 위상 맵 -> 학습에서는 정답만 사용!!!
        self.real_A =self.Phase_noise.to(self.device)
        
        inference = '이상치 마스크를 만들기 위해 추론 결과 영상을 저장할 경로'
        inference_root = 'inference가 위치하는 폴더'
        
        # fake_P: 모델이 복원한 real_A
        self.fake_P= self.netP(self.real_A)

        # fake_PB: 복원된 위상 맵인 fake_P를 위상 펼침을 통해 펼쳐진 위상 맵으로 변환
        self.fake_PB = self.fake_P[0].detach().cpu().numpy()
        self.fake_PB = np.angle(np.exp(1j * (self.fake_PB.astype(np.float32) * 2 * np.pi)))
        self.fake_PB = unwrap_phase(self.fake_PB)
        self.fake_PB = cv2.normalize(self.fake_PB, None, -1., 1., cv2.NORM_MINMAX)
        self.fake_PB = torch.Tensor(self.fake_PB)
        self.fake_PB = self.fake_PB.unsqueeze(0)
        self.fake_PB = self.fake_PB.to(self.device)
        zxcv = self.fake_PB.detach().cpu().numpy()
        asdf = cv2.normalize(zxcv[0], None, 0, 255, cv2.NORM_MINMAX)
        asdf = np.transpose(asdf, (1, 2, 0))
        
        # fake_PB 에서 여전히 존재하는 오차 영역에 대한 PAM을 통한 마스크 영역 생성
        # index: PCB1, PCB2, PCB3 이 각각 1000장씩 존재하며, 
        #   순서대로 영상 이름이 정렬되어 각 PCB 마다 다른 f-AnoGAN 모델 사용
        if self.index < 10:
            cv2.imwrite(inference,asdf)
            anomal_image = self.anomal_detect(test_root=inference_root, nnum = 1)
            zxcv = anomal_image.detach().cpu().numpy()
            asdf = cv2.normalize(zxcv[0], None, 0, 255, cv2.NORM_MINMAX)
            asdf = np.transpose(asdf, (1, 2, 0))
            maskk = self.anomal_mask(asdf)  

        if self.index < 20 and self.index>=10:
            cv2.imwrite(inference,asdf)
            anomal_image = self.anomal_detect(test_root=inference_root, nnum = 2)
            zxcv = anomal_image.detach().cpu().numpy()
            asdf = cv2.normalize(zxcv[0], None, 0, 255, cv2.NORM_MINMAX)
            asdf = np.transpose(asdf, (1, 2, 0))
            maskk = self.anomal_mask(asdf)  

        if self.index>=20:
            cv2.imwrite(inference,asdf)
            anomal_image = self.anomal_detect(test_root=inference_root, nnum = 3)
            zxcv = anomal_image.detach().cpu().numpy()
            asdf = cv2.normalize(zxcv[0], None, 0, 255, cv2.NORM_MINMAX)
            asdf = np.transpose(asdf, (1, 2, 0))
            maskk = self.anomal_mask(asdf) 
        
        maskk = torch.from_numpy(maskk).bool().to(self.device)

        self.Unwrap_noise.narrow(1,0,1).masked_fill_(maskk, 1)
        self.Unwrap_noise.narrow(1,1,1).masked_fill_(maskk, 1)
        self.Unwrap_noise.narrow(1,2,1).masked_fill_(maskk, 1)

        # real_C: 그림자와 빛 반사가 존재하는 펼쳐진 위상 맵(2D Unwrap) -> 학습에서는 정답만 사용!!!  
        self.real_C =self.Unwrap_noise.to(self.device)

        # fake_B: 복원된 real_C
        self.fake_B= self.netP2(self.real_C)
        self.real_B = self.Phase_real.to(self.device)
        self.real_D = self.Unwrap_real.to(self.device)

    def set_gt_latent(self):
        gt_latent=self.vgg(Variable(self.Phase_real,requires_grad=False))
        gt_latent2=self.vgg2(Variable(self.Unwrap_real,requires_grad=False))
        self.Cosis_list[0].set_target(gt_latent.relu4_3)
        self.Cosis_list2[0].set_target(gt_latent.relu4_3)
        self.Cosis_list2[0].set_target(gt_latent2.relu4_3)
        self.Cosis_list22[0].set_target(gt_latent2.relu4_3)

    # 모델 테스트 과정
    def test(self,i):
        # real_A: 그림자와 빛 반사가 존재하는 PCB 위상 맵
        self.real_A =self.Phase_noise.to(self.device)

        inference = '이상치 마스크를 만들기 위해 추론 결과 영상을 저장할 경로'
        inference_root = 'inference가 위치하는 폴더'
        
        self.real_C =self.Unwrap_noise.to(self.device)
        
        # fake_P: 복원된 real_A
        self.fake_P= self.netP(self.real_A)

        # fake_PB: 복원된 위상 맵인 fake_P를 위상 펼침을 통해 펼쳐진 위상 맵으로 변환
        self.fake_PB = self.fake_P[0].detach().cpu().numpy()
        self.fake_PB = np.angle(np.exp(1j * (self.fake_PB.astype(np.float32) * 2 * np.pi)))
        self.fake_PB = unwrap_phase(self.fake_PB)
        self.fake_PB = cv2.normalize(self.fake_PB, None, -1., 1., cv2.NORM_MINMAX)
        self.fake_PB = torch.Tensor(self.fake_PB)
        self.fake_PB = self.fake_PB.unsqueeze(0)
        self.fake_PB = self.fake_PB.to(self.device)
        zxcv = self.fake_PB.detach().cpu().numpy()
        asdf = cv2.normalize(zxcv[0], None, 0, 255, cv2.NORM_MINMAX)
        asdf = np.transpose(asdf, (1, 2, 0))

        # fake_PB 에서 여전히 존재하는 오차 영역에 대한 PAM을 통한 마스크 영역 생성
        # index: PCB1, PCB2, PCB3 이 각각 1000장씩 존재하며, 
        #   순서대로 영상 이름이 정렬되어 각 PCB 마다 다른 f-AnoGAN 모델 사용
        if self.index < 10:
            
            cv2.imwrite(inference,asdf)
            anomal_image = self.anomal_detect(test_root=inference_root, nnum = 1)
            zxcv = anomal_image.detach().cpu().numpy()
            asdf = cv2.normalize(zxcv[0], None, 0, 255, cv2.NORM_MINMAX)
            asdf = np.transpose(asdf, (1, 2, 0))
            maskk = self.anomal_mask(asdf)  

        if self.index < 20 and self.index>=10:
            cv2.imwrite(inference,asdf)
            anomal_image = self.anomal_detect(test_root=inference_root, nnum = 2)
            zxcv = anomal_image.detach().cpu().numpy()
            asdf = cv2.normalize(zxcv[0], None, 0, 255, cv2.NORM_MINMAX)
            asdf = np.transpose(asdf, (1, 2, 0))
            maskk = self.anomal_mask(asdf)  

        if self.index>=20:
            cv2.imwrite(inference,asdf)
            anomal_image = self.anomal_detect(test_root=inference_root, nnum = 3)
            zxcv = anomal_image.detach().cpu().numpy()
            asdf = cv2.normalize(zxcv[0], None, 0, 255, cv2.NORM_MINMAX)
            asdf = np.transpose(asdf, (1, 2, 0))
            maskk = self.anomal_mask(asdf) 
        
        maskk = torch.from_numpy(maskk).bool().to(self.device)
        
        self.fake_PB.narrow(1,0,1).masked_fill_(maskk, 1)
        self.fake_PB.narrow(1,1,1).masked_fill_(maskk, 1)
        self.fake_PB.narrow(1,2,1).masked_fill_(maskk, 1)

        # fake_B: 앞선 학습과 달리 첫 스테이지에서 복원되어 펼쳐진 위상 맵으로 변환된 fake_PB를 복원
        self.fake_B= self.netP2(self.fake_PB)
        self.real_B = self.Phase_real.to(self.device)
        self.real_D = self.fake_PB.to(self.device)


    # 판별자 갱신
    def backward_D(self):
        
        fake_AB = self.fake_P
        
        self.gt_latent_fake = self.vgg(Variable(self.fake_P.data, requires_grad=False))
        self.gt_latent_real = self.vgg(Variable(self.real_B, requires_grad=False))
        real_AB = self.real_B 

        # 판별자 D에서 판별한 가짜와 실제 간의 loss 계산
        self.pred_fake = self.netD(fake_AB.detach())
        self.pred_real = self.netD(real_AB)
        self.loss_D_fake = self.criterionGAN(self.pred_fake, self.pred_real, True)

        # 판별자 F에서 판별한 가짜와 실제 간의 loss 계산
        self.pred_fake_F = self.netF(self.gt_latent_fake.relu3_3.detach())
        self.pred_real_F = self.netF(self.gt_latent_real.relu3_3)
        self.loss_F_fake = self.criterionGAN(self.pred_fake_F,self.pred_real_F, True)

        self.loss_D =self.loss_D_fake * 0.5 + self.loss_F_fake  * 0.5

        self.loss_D.backward()

    # 생성자 갱신
    def backward_G(self):

        fake_AB = self.fake_P
        self.gt_latent_fake = self.vgg(Variable(self.fake_P.data, requires_grad=False))
        self.gt_latent_real = self.vgg(Variable(self.real_B, requires_grad=False))
        fake_f = self.gt_latent_fake
        
        # 본 논문에서는 언급하지 않았던 모아레 특성상 펼쳐진 위상 맵의 높이 차이는 같아야 한다는 점을 고려하여 loss 설계
        loss_GEN = height_difference_loss(fake_AB)
        
        grad_mean, entropy_mean, std_dev_mean = Uniform_loss(fake_AB.detach())
        self.loss_Grad = float(grad_mean)*1000
        self.loss_Entropy = float(entropy_mean)*500
        self.loss_Std_dev = float(std_dev_mean)*1200
        
        self.pred_fake = self.netD(fake_AB)
        self.pred_fake_f = self.netF(fake_f.relu3_3)
        
        pred_real=self.netD(self.real_B)
        pred_real_F=self.netF(self.gt_latent_real2.relu3_3)

        # 생성한 가짜와 정답간의 loss 계산
        self.loss_G_GAN = self.criterionGAN(self.pred_fake,pred_real, False)+self.criterionGAN(self.pred_fake_f, pred_real_F,False)
        self.loss_G_L1 =( self.criterionL1(self.fake_P, self.real_B) +self.criterionL1(self.fake_P, self.real_B) )* self.opt.lambda_A
        self.loss_height = float(loss_GEN)
        self.loss_G = self.loss_G_L1 + self.loss_G_GAN * self.opt.gan_weight + self.loss_height + self.loss_Grad + self.loss_Entropy + self.loss_Std_dev
        self.loss_G.backward()
        
        return self.loss_Grad , self.loss_Entropy , self.loss_Std_dev, self.loss_G_GAN , self.loss_G_L1, self.loss_D_fake, self.loss_F_fake

    def backward_D2(self):
        
        fake_AB2 = self.fake_B
        # Real
        self.gt_latent_fake2 = self.vgg2(Variable(self.fake_B.data, requires_grad=False))
        self.gt_latent_real2 = self.vgg2(Variable(self.real_D, requires_grad=False))
        real_AB2 = self.real_D # GroundTruth

        self.pred_fake2 = self.netD2(fake_AB2.detach())
        self.pred_real2 = self.netD2(real_AB2)
        self.loss_D_fake2 = self.criterionGAN2(self.pred_fake2, self.pred_real2, True)

        self.pred_fake_F2 = self.netF2(self.gt_latent_fake2.relu3_3.detach())
        self.pred_real_F2 = self.netF2(self.gt_latent_real2.relu3_3)
        self.loss_F_fake2 = self.criterionGAN2(self.pred_fake_F2,self.pred_real_F2, True)

        self.loss_D2 =self.loss_D_fake2 * 0.5 + self.loss_F_fake2  * 0.5

        self.loss_D2.backward()

    

    def backward_G2(self):
        
        fake_AB = self.fake_B
        self.gt_latent_fake2 = self.vgg2(Variable(self.fake_B.data, requires_grad=False))
        self.gt_latent_real2 = self.vgg2(Variable(self.real_D, requires_grad=False))
        fake_f2 = self.gt_latent_fake2

        loss_GEN = height_difference_loss(fake_AB)
        
        
        grad_mean, entropy_mean, std_dev_mean = Uniform_loss(fake_AB.detach())
        self.loss_Grad2 = float(grad_mean)*1000
        self.loss_Entropy2 = float(entropy_mean)*500
        self.loss_Std_dev2 = float(std_dev_mean)*1200
        
        self.pred_fake2 = self.netD2(fake_AB)
        self.pred_fake_f2 = self.netF2(fake_f2.relu3_3)
        
        pred_real2=self.netD2(self.real_D)
        pred_real_F2=self.netF2(self.gt_latent_real2.relu3_3)

        self.loss_G_GAN2 = self.criterionGAN2(self.pred_fake2,pred_real2, False)+self.criterionGAN2(self.pred_fake_f2, pred_real_F2,False)
        self.loss_G_L12 =( self.criterionL12(self.fake_B, self.real_D) +self.criterionL12(self.fake_B, self.real_D) )* self.opt.lambda_A
        self.loss_height2 = float(loss_GEN)
        self.loss_G2 = self.loss_G_L12 + self.loss_G_GAN2 * self.opt.gan_weight + self.loss_height2 + self.loss_Grad2 + self.loss_Entropy2 + self.loss_Std_dev2
        self.loss_G2.backward()
        return self.loss_Grad2 , self.loss_Entropy2 , self.loss_Std_dev2, self.loss_G_GAN2 , self.loss_G_L12, self.loss_D_fake2, self.loss_F_fake2

    # 파라미터 갱신 전체 플로우
    def optimize_parameters(self):
        # 학습 진행
        self.forward()
        # 각각의 생성자, 판별자 초기화
        self.optimizer_D.zero_grad()
        self.optimizer_F.zero_grad()
        self.optimizer_D2.zero_grad()
        self.optimizer_F2.zero_grad()
        # 학습을 토대로 각각의 생성자, 판별자 갱신
        self.backward_D()
        self.backward_D2()
        self.optimizer_D.step()
        self.optimizer_F.step()
        self.optimizer_D2.step()
        self.optimizer_F2.step()
        self.optimizer_P2.zero_grad()
        self.optimizer_P.zero_grad()
        # 손실 값 계산
        self.loss_Grad2 , self.loss_Entropy2 , self.loss_Std_dev2, self.loss_G_GAN2 , self.loss_G_L12, self.loss_D_fake2, self.loss_F_fake2 = self.backward_G2()
        self.loss_Grad , self.loss_Entropy , self.loss_Std_dev, self.loss_G_GAN , self.loss_G_L1, self.loss_D_fake, self.loss_F_fake = self.backward_G()
        self.optimizer_P2.step()
        self.optimizer_P.step()

        return self.loss_Grad2 , self.loss_Entropy2 , self.loss_Std_dev2, self.loss_G_GAN2 , self.loss_G_L12, self.loss_D_fake2, self.loss_F_fake2

    # 학습 출력 로그
    def get_current_errors(self):
        return OrderedDict([
        ('Grad_loss',self.loss_Grad2),
        ('Entropy_loss', self.loss_Entropy2),
        ('Std_dev_loss', self.loss_Std_dev2),
        ('G_GAN', self.loss_G_GAN2.data.item()),
                            ('G_L1', self.loss_G_L12.data.item()),
                            ('D', self.loss_D_fake2.data.item()),
                            ('F', self.loss_F_fake2.data.item())
                            ])

    # 학습/테스트 결과 시각화
    def get_current_visuals(self):

        real_A =self.real_A.data
        real_C =self.real_C.data
        fake_B = self.fake_B.data
        fake_P = self.fake_P.data
        real_B =self.real_B.data
        real_D =self.real_D.data

        return real_A,real_B,fake_P, real_C, real_D, fake_B

    # 각각의 판별자, 생성자 저장
    def save(self, epoch):
        self.save_network(self.netP2, 'P2', epoch, self.gpu_ids)
        self.save_network(self.netP, 'P', epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', epoch, self.gpu_ids)
        self.save_network(self.netF, 'F', epoch, self.gpu_ids)
        self.save_network(self.netD2, 'D2', epoch, self.gpu_ids)
        self.save_network(self.netF2, 'F2', epoch, self.gpu_ids)

    # 테스트 과정에서는 생성자만 로드
    def load(self, epoch):
        self.load_network(self.netP2, 'P2', epoch)
        self.load_network(self.netP, 'P', epoch)