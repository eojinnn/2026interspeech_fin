import torch
import torch.nn as nn

from .dnn_models import SincNet

from .resnet import resnet18, resnet18_nopool, BasicBlock
from .conformer import ConformerBlock

import numpy as np

layer_resnet = ['conv1', 'bn1', 'relu', 'layer1', 'layer1.0', 'layer1.0.conv1', 'layer1.0.bn1', 'layer1.0.relu', 'layer1.0.conv2', 'layer1.0.bn2', 'layer1.1', 'layer1.1.conv1', 'layer1.1.bn1', 'layer1.1.relu', 'layer1.1.conv2', 'layer1.1.bn2', 'maxpool1', 'layer2', 'layer2.0', 'layer2.0.conv1', 'layer2.0.bn1', 'layer2.0.relu', 'layer2.0.conv2', 'layer2.0.bn2', 'layer2.0.downsample', 'layer2.0.downsample.0', 'layer2.0.downsample.1', 'layer2.1', 'layer2.1.conv1', 'layer2.1.bn1', 'layer2.1.relu', 'layer2.1.conv2', 'layer2.1.bn2', 'maxpool2', 'layer3', 'layer3.0', 'layer3.0.conv1', 'layer3.0.bn1', 'layer3.0.relu', 'layer3.0.conv2', 'layer3.0.bn2', 'layer3.0.downsample', 'layer3.0.downsample.0', 'layer3.0.downsample.1', 'layer3.1', 'layer3.1.conv1', 'layer3.1.bn1', 'layer3.1.relu', 'layer3.1.conv2', 'layer3.1.bn2', 'maxpool3', 'layer4', 'layer4.0', 'layer4.0.conv1', 'layer4.0.bn1', 'layer4.0.relu', 'layer4.0.conv2', 'layer4.0.bn2', 'layer4.0.downsample', 'layer4.0.downsample.0', 'layer4.0.downsample.1', 'layer4.1', 'layer4.1.conv1', 'layer4.1.bn1', 'layer4.1.relu', 'layer4.1.conv2', 'layer4.1.bn2', 'conv5']

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class GCC_PHAT(nn.Module):
    def __init__(self, n_lags=64):
        super().__init__()
        self.n_lags = n_lags # 보통 64나 128 정도만 씁니다 (최대 지연 범위)

    def forward(self, x):
        # x shape: [Batch, Mics(4), Frames(500), Win_Len(480)]
        B, M, T, L = x.shape
        
        # 1. FFT 수행 (Real-input FFT) -> 주파수 도메인으로 변환
        # 결과 shape: [B, M, T, L//2 + 1] (복소수)
        X = torch.fft.rfft(x, dim=-1)
        
        cc_list = []
        # 2. 마이크 쌍(Pair)에 대해 Correlation 계산 (6개 조합)
        # (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        for m1 in range(M):
            for m2 in range(m1+1, M):
                # Cross Power Spectrum: X1 * conj(X2)
                X1 = X[:, m1, :, :]
                X2 = X[:, m2, :, :]
                R = X1 * torch.conj(X2)
                
                # PHAT Weighting: 크기로 나누어 정규화 (위상 정보만 남김)
                # 0으로 나누는 것 방지 (eps)
                R = R / (torch.abs(R) + 1e-8)
                
                # 3. IFFT 수행 -> 시간 도메인(Delay)으로 변환 (GCC)
                cc = torch.fft.irfft(R, dim=-1)
                
                # 4. Lag 자르기 (Shift & Crop)
                # IFFT 결과는 [0, 1, ... , -2, -1] 순서이므로
                # 이를 [-tau, ..., 0, ..., +tau] 순서로 바꾸고 필요한 만큼만 자름
                cc = torch.fft.fftshift(cc, dim=-1)
                
                # 중앙을 기준으로 n_lags 만큼만 가져오기
                center = L // 2
                start = center - self.n_lags // 2
                end = start + self.n_lags
                cc = cc[:, :, start:end] # [Batch, Frames, n_lags]
                
                cc_list.append(cc)

        # 5. 채널로 쌓기 (Stack)
        # 결과: [Batch, Pairs(6), Frames(500), n_lags]
        out = torch.stack(cc_list, dim=1)
        
        return out

class PostBackboneCNN(nn.Module):
    def __init__(self, in_channels=64, target_frames=500, origin_samples=240000):
        super().__init__()
        
        # stride 계산: 240000 / 500 = 480
        stride_len = origin_samples // target_frames
        kernel_len = stride_len  # 겹치지 않게 하거나, 겹치게 하려면 2*stride_len 등 설정
        
        # 1. 1D CNN for Time Downsampling
        # 입력: 64채널 (SincNet feature)
        # 출력: 64채널 (유지하거나 변경 가능)
        # 역할: Time 축을 1/480로 압축
        self.time_conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=kernel_len, stride=stride_len, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU()
        )

    def forward(self, x):
        # x input from backbone: [64, 64, 240000] (Batch*Mics, Feat, Time)
        
        # 1. 사용자가 원하는 형태로 복구 (Shape Restore)
        # [Batch*Mics, C, T] -> [Batch, Mics, C, T]
        # 예: [64, 64, 240000] -> [16, 4, 64, 240000]
        BM, C, T = x.shape
        B = BM // 4  # Mics=4라고 가정
        M = 4
        x = x.view(B, M, C, T)
        # 2. 1D CNN 적용을 위해 다시 병합 (Apply 1D CNN)
        # Conv1d는 3D 텐서(N, C, L)를 원하므로 B와 M을 합침
        x = x.view(B * M, C, T)
        
        # Time Downsampling: [64, 64, 240000] -> [64, 64, 500]
        x = self.time_conv(x)
        
        # 3. 마이크 차원 확장 (Mic Mixing)
        # 다시 분리: [64, 64, 500] -> [16, 4, 64, 500]
        _, C_new, T_new = x.shape
        x = x.view(B, M, C_new, T_new)
        
        # 4. 최종 출력 형태 정리
        # [Batch, Channels, Time, Feat] 형태로 변경 (모델 뒷단 요구사항에 따라)
        # [16, 4, 64, 500] -> [16, 4, 500, 64]
        x = x.permute(0, 1, 3, 2)
        
        return x
    
    
class ResnetConformer_seddoa_nopool_2023(nn.Module):
    def __init__(self, in_channel, in_dim, out_dim, n_mel_bins=64, use_sinc=True,
                                        sig_len=480, num_channels=32, fs=24000, normalize_output=False, pool_len=5,):
        super().__init__()
        self.resnet = resnet18_nopool(in_channel=in_channel)
        embedding_dim = in_dim // 32 * 256
        encoder_dim = 256
        self.input_projection = nn.Sequential(
            nn.Linear(embedding_dim, encoder_dim),
            nn.Dropout(p=0.05),
        )
        num_layers = 8
        self.conformer_layers = nn.ModuleList(
            [ConformerBlock(
                dim = encoder_dim,
                dim_head = 32,
                heads = 8,
                ff_mult = 2,
                conv_expansion_factor = 2,
                conv_kernel_size = 7,
                attn_dropout = 0.1,
                ff_dropout = 0.1,
                conv_dropout = 0.1
            ) for _ in range(num_layers)]
        )
        self.t_pooling = nn.MaxPool1d(kernel_size=5)
        self.sed_out_layer = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.LeakyReLU(),
            nn.Linear(encoder_dim, 13),
            nn.Sigmoid()
        )
        self.out_layer = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.LeakyReLU(),
            nn.Linear(encoder_dim, out_dim),
            nn.Tanh()
        ) 
        sincnet_params = {'input_dim': sig_len,
                    'fs': fs,
                    'cnn_N_filt': [num_channels, num_channels, num_channels, in_dim],
                    'cnn_len_filt': [sig_len-1, 11, 9, 7],
                    'cnn_max_pool_len': [1,1,1,1],
                    'cnn_use_laynorm_inp': False,
                    'cnn_use_batchnorm_inp': False,
                    'cnn_use_laynorm': [False, False, False, False],
                    'cnn_use_batchnorm': [True, True, True, True],
                    'cnn_act': ['leaky_relu', 'leaky_relu', 'leaky_relu', 'linear'],
                    'cnn_drop': [0.0, 0.0, 0.0, 0.0],
                    'use_sinc': use_sinc,
                    } 

        self.backbone = SincNet(sincnet_params)
        self.post_process = PostBackboneCNN(in_channels=64)

    def forward(self, audio):
        # pdb.set_trace() # [32, 7, 500, 64]
        B, M, T, L = audio.shape # (batch_size, #mics, #time_windows, win_len)
        x = audio.reshape(-1, 1, T*L)
        x = self.backbone(x) #torch.Size([64, 64, 240000])
        x = self.post_process(x) (16, 4, 500, 64)
        

        conv_outputs = self.resnet(x)
        N,C,T,W = conv_outputs.shape
        conv_outputs = conv_outputs.permute(0,2,1,3).reshape(N, T, C*W)

        conformer_outputs = self.input_projection(conv_outputs)
        #conformer_outputs = conv_outputs
        for layer in self.conformer_layers:
            conformer_outputs = layer(conformer_outputs)
        outputs = conformer_outputs.permute(0,2,1)
        outputs = self.t_pooling(outputs)
        outputs = outputs.permute(0,2,1)
        sed = self.sed_out_layer(outputs)
        doa = self.out_layer(outputs)
        # dist = self.dist_out_layer(outputs)
        pred = torch.cat((sed, doa), dim=-1) # [32, 100, 52]
        # pdb.set_trace()
        return pred