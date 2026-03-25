import torch
import math
import joblib
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

class GlobalScalar(nn.Module):
    def __init__(self, scaler_path):
        super().__init__()
        scaler = joblib.load(scaler_path)
        mean = torch.tensor(scaler.mean_, dtype=torch.float32).view(1,10,1,64)
        std = torch.tensor(scaler.scale_, dtype=torch.float32).view(1,10,1,64)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
    
    def forward(self, x):
        return (x - self.mean) / (self.std + 1e-8)

class AuditoryFrontend(nn.Module):
    def __init__(self, sr=24000, n_bins=64, n_lags=64, hop_length=480, win_length=960, n_fft=1024, scaler_path=None):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_lags = n_lags
        self.n_freqs = n_fft // 2 + 1
        self.init_slope = 10.0
        self.init_transition_ratio = 0.5
        
        # 멜 필터뱅크 (torchaudio 기능 활용하여 필터 행렬만 가져옴)
        mel_filters = torchaudio.functional.melscale_fbanks(
            int(n_fft // 2 + 1), n_mels=n_bins, f_min=0.0, f_max=sr/2, sample_rate=sr, norm='slaney'
        )
        self.register_buffer('mel_filters', mel_filters)
        self.register_buffer('window', torch.hann_window(win_length))

        freq_axis = torch.linspace(0.0, 1.0, steps=self.n_freqs)
        # psychoacoustic prior
        # mel: low 작고 high 큼
        mel_init = torch.sigmoid(self.init_slope * (freq_axis - self.init_transition_ratio))
        # gcc: low 큼고 high 작음
        gcc_init = 1.0 - mel_init

        eps = 1e-4
        mel_init = mel_init.clamp(eps, 1.0 - eps)
        gcc_init = gcc_init.clamp(eps, 1.0 - eps)

        mel_logits = torch.log(mel_init / (1.0 - mel_init))
        gcc_logits = torch.log(gcc_init / (1.0 - gcc_init))

        self.mel_gate_logits = nn.Parameter(mel_logits)
        self.gcc_gate_logits = nn.Parameter(gcc_logits)
        
        self.scaler = GlobalScalar(scaler_path)
    
    def get_weights(self):
        mel_weight = torch.sigmoid(self.mel_gate_logits)
        gcc_weight = torch.sigmoid(self.gcc_gate_logits)
        return mel_weight, gcc_weight

    def forward(self, wav):
        B, M, F, T = wav.shape
        wav = wav.reshape(B * M, F*T)
        
        # 1. 공통 STFT 계산 (기존 librosa와 동일한 환경)
        stft = torch.stft(wav, n_fft=self.n_fft, hop_length=self.hop_length, 
                          win_length=self.win_length, window=self.window, 
                          center=True, return_complex=True)
        # stft shape: [B*M, Freq, Frames]
        stft = stft.reshape(B, M, stft.shape[1], stft.shape[2]) # [B, M, Freq, Frames]
        if stft.shape[-1] > F:
            stft = stft[:, :, :, :F]
        
        mel_weight, gcc_weight = self.get_weights()
        mel_weight = mel_weight.to(stft.device)
        gcc_weight = gcc_weight.to(stft.device)

        # 2. Mel Spectrogram 추출
        mag = torch.abs(stft)
        mel_w = mel_weight.view(1, 1, -1, 1)
        power = (mag ** 2) * mel_w
        mel = torch.matmul(power.transpose(-1, -2), self.mel_filters).transpose(-1, -2)
        mel = 10.0 * torch.log10(mel.clamp_min(1e-10))
        # [B, M, n_mels, Frames]
        max_val = mel.amax(dim=(-2,-1), keepdim=True)
        mel = torch.maximum(mel, max_val - 80.0)
        
        # 3. GCC-PHAT 추출
        cc_list = []
        gcc_w = gcc_weight.view(1, -1, 1)  # [1, Freq, 1]
        for m1 in range(M):
            for m2 in range(m1 + 1, M):
                X1 = stft[:, m1, :, :]
                X2 = stft[:, m2, :, :]
                R = X1 * torch.conj(X2)
                R_phat = R / (torch.abs(R) + 1e-8) # PHAT Weighting
                R_weighted = R_phat * gcc_w
                
                # IFFT로 lag 도메인 변환
                cc = torch.fft.irfft(R_weighted, n=self.n_fft, dim=1)
                cc = torch.fft.fftshift(cc, dim=1) # 중앙 정렬
                
                # 중앙에서 n_lags 만큼 Crop
                center = self.n_fft // 2
                start = center - self.n_lags // 2
                end = start + self.n_lags
                cc_cropped = cc[:, start:end, :] # [B, n_lags, Frames]
                cc_list.append(cc_cropped)
                
        gcc = torch.stack(cc_list, dim=1) # [B, 6, n_lags, Frames]
        
        # 4. Mel과 GCC 결합 (채널 차원)
        out = torch.cat([mel, gcc], dim=1) # [B, 10, Feature, Frames]
        out = out.permute(0, 1, 3, 2) # 모델 입력에 맞게 [Batch, Channel, Time, Feature]로 변환
        out = self.scaler(out)
        
        return out

class ResnetConformer_seddoa_nopool_2023(nn.Module):
    def __init__(self, in_channel, in_dim, out_dim, n_mel_bins=64, sig_len=480, fs=24000, norm_path=None):
        super().__init__()
        self.resnet = resnet18_nopool(in_channel=in_channel)
        embedding_dim = in_dim // 32 * 256
        encoder_dim = 256
        self.input_projection = nn.Sequential(
            nn.Linear(embedding_dim, encoder_dim),
            nn.Dropout(p=0.05),
        )
        num_layers = 8
        self.frontend = AuditoryFrontend(scaler_path=norm_path)
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