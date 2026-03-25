import torch
import math
import torchaudio
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .resnet import resnet18_nopool
from .conformer import ConformerBlock
from nnAudio.features.gammatone import Gammatonegram


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
        X = torch.fft.rfft(x, dim=-1)
        
        cc_list = []
        # 2. 마이크 쌍(Pair)에 대해 Correlation 계산 (6개 조합)
        for m1 in range(M):
            for m2 in range(m1+1, M):
                # Cross Power Spectrum: X1 * conj(X2)
                X1 = X[:, m1, :, :]
                X2 = X[:, m2, :, :]
                R = X1 * torch.conj(X2)
                
                # PHAT Weighting: 크기로 나누어 정규화 (위상 정보만 남김)
                R = R / (torch.abs(R) + 1e-8)
                
                # 3. IFFT 수행 -> 시간 도메인(Delay)으로 변환 (GCC)
                cc = torch.fft.irfft(R, dim=-1)
                
                # 4. Lag 자르기 (Shift & Crop)
                cc = torch.fft.fftshift(cc, dim=-1)
                
                center = L // 2
                start = center - self.n_lags // 2
                end = start + self.n_lags
                cc = cc[:, :, start:end] # [Batch, Frames, n_lags]
                
                cc_list.append(cc)

        # 5. 채널로 쌓기 (Stack)
        # 결과: [Batch, Pairs(6), Frames(500), n_lags]
        out = torch.stack(cc_list, dim=1)
        
        return out

class GammatoneGCCFrontend(nn.Module):
    """
    Gammatone Filterbank와 GCC-PHAT를 병렬로 추출하여 채널 차원으로 결합하는 프론트엔드 모듈
    """
    def __init__(self, sr=24000, n_bins=64, n_lags=64, hop_length=480, fmin=50, fmax=8000):
        super().__init__()
        self.gcc = GCC_PHAT(n_lags=n_lags)
        self.n_mels = n_bins
        self.hop_length = hop_length
        self.win_length = hop_length * 2
        self.n_fft = 2 ** (self.win_length - 1).bit_length()
        self.fmax = fmax
        
        # Gammatone 필터뱅크 행렬 가중치 (학습 가능하게 설정됨)
        # ※ 주의: 실제 정통 Gammatone 형태를 띄게 하려면 모델 초기화 직후에 
        # python 라이브러리(예: spafe)로 생성한 weight를 이 파라미터에 덮어씌우고
        # requires_grad=False 처리하는 방식을 권장합니다.
        self.gt = Gammatonegram(sr = sr, n_fft=self.n_fft, n_bins=n_bins, hop_length=self.hop_length, window='hann', power = 2, fmax=self.fmax, win_length=self.win_length, trainable_STFT=False, trainable_bins=False)
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, n_mels=self.n_mels, window_fn=torch.hann_window)

    def forward(self, x):
        # x shape: [Batch, Mics(4), Frames(T), Win_Len(L)]
        
        # 1. Gammatone Spectrogram 추출
        # rfft를 통해 크기(Magnitude)를 구함
        B, M, T, L = x.shape

        gcc_feat = self.gcc(x)
        x = x.reshape(B, M, T * L)
        x = x.reshape(B * M, T * L)
        # feat = self.gt(x) # [Batch*Mics, n_mels, Frames]
        feat = self.mel(x)
        feat = 10.0 * torch.log10(feat.clamp_min(1e-10))
        feat = feat.reshape(B, M, feat.shape[-2], feat.shape[-1])
        feat = feat.transpose(-2, -1)  # [B, M, T, F] 맞추기
        if feat.shape[2] > T:
            feat = feat[:, :, :T, :]

        out = torch.cat([feat, gcc_feat], dim=1)
        return out

class ResnetConformer_seddoa_nopool_2023(nn.Module):
    def __init__(self, in_channel, in_dim, out_dim, n_mel_bins=64, sig_len=480, fs=24000):
        super().__init__()
        self.resnet = resnet18_nopool(in_channel=in_channel)
        embedding_dim = in_dim // 32 * 256
        encoder_dim = 256
        self.input_projection = nn.Sequential(
            nn.Linear(embedding_dim, encoder_dim),
            nn.Dropout(p=0.05),
        )
        num_layers = 8
        self.frontend = GammatoneGCCFrontend()
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

    def forward(self, x):
        B, M, T, L = x.shape
        x = self.frontend(x) # (B, 4, 64, 500)
        # pdb.set_trace() # [32, 7, 500, 64]
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