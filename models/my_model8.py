import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .resnet import resnet18_nopool
from .conformer import ConformerBlock


class LearnableSincLowpass(nn.Module):
    """
    SincNet-style learnable lowpass FIR.
    - Input : (B, C, T)
    - Output: (B, C, T)  (same length by padding)
    """
    def __init__(
        self,
        channels: int,
        kernel_size: int = 63,
        init_cutoff_hz: float = 6000.0,
        sample_rate: int = 24000,
        per_channel: bool = False,
        min_cutoff_hz: float = 50.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size should be odd."
        self.channels = channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.per_channel = per_channel
        self.min_cutoff_hz = min_cutoff_hz
        self.eps = eps

        # learnable parameter in unconstrained space -> sigmoid -> (min, nyquist)
        nyq = sample_rate / 2.0
        init = (init_cutoff_hz - min_cutoff_hz) / (nyq - min_cutoff_hz + eps)
        init = float(min(max(init, 1e-4), 1 - 1e-4))
        init_logit = math.log(init / (1.0 - init))

        n_params = channels if per_channel else 1
        self.cutoff_logit = nn.Parameter(torch.full((n_params,), init_logit))

        # time index centered at 0
        M = (kernel_size - 1) // 2
        t = torch.arange(-M, M + 1).float()  # (K,)
        self.register_buffer("t", t)

        # Hamming window
        w = 0.54 - 0.46 * torch.cos(2 * math.pi * (torch.arange(kernel_size).float() / (kernel_size - 1)))
        self.register_buffer("window", w)

    def _make_kernel(self, cutoff_hz: torch.Tensor) -> torch.Tensor:
        # cutoff_hz: (1,) or (C,)
        # normalized cutoff (0..0.5) in cycles/sample
        fc = cutoff_hz / self.sample_rate  # (..,)
        t = self.t.to(cutoff_hz.device)  # (K,)

        # sinc lowpass: h[n] = 2*fc*sinc(2*fc*n)
        # torch.sinc(x) = sin(pi x)/(pi x)
        # so use torch.sinc(2*fc*t) where argument is in "cycles" scaled by pi inside sinc
        # We need sinc(2*fc*n) with sinc defined as sin(pi x)/(pi x) -> torch.sinc does that.
        # h = 2*fc * sinc(2*fc*t)
        # For vector fc, broadcast to (C,K)
        if fc.ndim == 1 and fc.numel() > 1:
            fc = fc[:, None]  # (C,1)
        else:
            fc = fc.view(1, 1)  # (1,1)

        h = 2.0 * fc * torch.sinc(2.0 * fc * t[None, :])  # (C or 1, K)
        h = h * self.window[None, :].to(h.device)

        # normalize DC gain to 1
        h = h / (h.sum(dim=-1, keepdim=True) + self.eps)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,T)
        nyq = self.sample_rate / 2.0
        s = torch.sigmoid(self.cutoff_logit)

        cutoff_hz = self.min_cutoff_hz + s * (nyq - self.min_cutoff_hz)
        h = self._make_kernel(cutoff_hz)  # (1,K) or (C,K)

        pad = self.kernel_size // 2

        if self.per_channel:
            # depthwise conv: (C,1,K), groups=C
            weight = h[:, None, :]  # (C,1,K)
            return F.conv1d(x, weight, padding=pad, groups=self.channels)
        else:
            # shared kernel across channels: apply depthwise with same weight repeated
            weight = h.repeat(self.channels, 1).unsqueeze(1)  # (C,1,K)
            return F.conv1d(x, weight, padding=pad, groups=self.channels)


# -------------------------
# ERB utilities (for gammatone center freqs)
# -------------------------
def hz_to_erb_number(f_hz: np.ndarray) -> np.ndarray:
    return 21.4 * np.log10(1.0 + 0.00437 * f_hz)

def erb_number_to_hz(erb: np.ndarray) -> np.ndarray:
    return (10 ** (erb / 21.4) - 1.0) / 0.00437

def erb_bandwidth(f_hz: np.ndarray) -> np.ndarray:
    return 24.7 * (4.37 * f_hz / 1000.0 + 1.0)

class FixedGammatoneFIR(nn.Module):
    """
    Fixed FIR gammatone filterbank.
    Input:  (B, 1, N)
    Output: (B, C, N)
    """
    def __init__(
        self,
        sample_rate: int = 24000,
        num_filters: int = 64,
        min_freq: float = 50.0,
        max_freq: float | None = None,
        kernel_size: int = 256,
        order_n: int = 4,
        scale_b: float = 1.019,
    ):
        super().__init__()
        self.fs = sample_rate
        self.C = num_filters
        self.kernel_size = int(kernel_size)
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1

        max_freq = float(max_freq) if max_freq is not None else (sample_rate / 2.0)
        erb_lo = hz_to_erb_number(np.array([min_freq]))[0]
        erb_hi = hz_to_erb_number(np.array([max_freq]))[0]
        erb_points = np.linspace(erb_lo, erb_hi, num_filters)
        center_freqs = erb_number_to_hz(erb_points)  # (C,)
        bws = erb_bandwidth(center_freqs)            # (C,)
        b = scale_b * bws

        t = np.arange(self.kernel_size, dtype=np.float64) / self.fs  # [0..K-1]/fs

        kernels = []
        for f0, bi in zip(center_freqs, b):
            env = (t ** (order_n - 1)) * np.exp(-2.0 * np.pi * bi * t)
            carrier = np.cos(2.0 * np.pi * f0 * t)
            h = env * carrier
            h = h / (np.linalg.norm(h) + 1e-12)
            kernels.append(h.astype(np.float32))

        kernels = np.stack(kernels, axis=0)              # (C, K)
        weight = torch.from_numpy(kernels).unsqueeze(1)  # (C,1,K)
        self.register_buffer("weight", weight)
        self.register_buffer("center_freqs_hz", torch.tensor(center_freqs, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,1,N) -> (B,C,N)
        w = self.weight.to(dtype=x.dtype, device=x.device)
        pad = self.kernel_size // 2
        return F.conv1d(x, w, stride=1, padding=pad)
    

import torch
import torch.nn as nn

class MapTo10_F64_T100_CNN2D(nn.Module):
    """
    Input : (B, 4, 64, 1500)
    Output: (B, 10, 64, 100)
    
    개선사항: 
    데이터를 버리지 않고(Aliasing 방지), stride 5와 stride 3을 순차적으로 적용하여 
    시간 축을 1500 -> 300 -> 100으로 안전하게 압축합니다.
    """
    def __init__(self, hidden=10, leak=0.2):
        super().__init__()
        self.act = nn.LeakyReLU(leak, inplace=True)
        
        # Step 1: 시간축 5배 압축 (1500 -> 300)
        # kernel_size의 Time 축을 stride와 동일한 5로 설정하여 버려지는 프레임이 없도록 함.
        self.conv_down1 = nn.Conv2d(
            in_channels=4, out_channels=32,
            kernel_size=(3, 5), stride=(1, 5), 
            padding=(1, 0), bias=False
        )
        self.bn1 = nn.BatchNorm2d(32)
        
        # Step 2: 시간축 3배 압축 (300 -> 100)
        # kernel_size의 Time 축을 3으로 설정하여 나머지 압축 진행. 최종 출력 채널(10)로 맞춤.
        self.conv_down2 = nn.Conv2d(
            in_channels=32, out_channels=hidden,
            kernel_size=(3, 3), stride=(1, 3), 
            padding=(1, 0), bias=False
        )
        self.bn2 = nn.BatchNorm2d(hidden)

    def forward(self, x):
        # x: (B, 4, 64, 1500)
        
        # 1차 압축: 1500 / 5 = 300
        h = self.conv_down1(x)
        h = self.act(self.bn1(h))  # (B, 16, 64, 300)
        
        # 2차 압축: 300 / 3 = 100
        h = self.conv_down2(h)
        y = self.act(self.bn2(h))  # (B, 10, 64, 100)
        
        return y.contiguous()

    
class DownConvDecimate5x(nn.Module):
    """
    Input : (B, 4, 48000)
    Step  : reshape -> (B*4, 48000, 1)  (채널=1, 길이=48000)
            then 5x [conv1d(stride=1) -> decimate(::2)]
    Output: (B, 4, 1500, 32)
    """
    def __init__(self, out_ch=64, kernel_size=15, padding="same", leak=0.2,sample_rate=24000, num_filters=64, gammatone_kernel=1024, eps=1e-8):
        super().__init__()
        assert padding in ["same", "valid"]

        self.C = num_filters
        self.eps = eps
        self.gammatone = FixedGammatoneFIR(
            sample_rate=sample_rate,
            num_filters=num_filters,
            min_freq=50.0,
            # max_freq=sample_rate / 2.0,
            max_freq=8000.0,  
            kernel_size=gammatone_kernel,
            order_n=4,
            scale_b=1.019,
        )
        self.fs0 = sample_rate
        self.aa0 = LearnableSincLowpass(
            channels=num_filters, kernel_size=63,
            init_cutoff_hz=5000.0, sample_rate=self.fs0,
            per_channel=False
        )
        self.aa1 = LearnableSincLowpass(64, 63, init_cutoff_hz=2500.0, sample_rate=self.fs0//2,  per_channel=False)
        self.aa2 = LearnableSincLowpass(64, 63, init_cutoff_hz=1250.0, sample_rate=self.fs0//4,  per_channel=False)
        self.aa3 = LearnableSincLowpass(64, 63, init_cutoff_hz=600.0, sample_rate=self.fs0//8,  per_channel=False)
        self.aa4 = LearnableSincLowpass(64, 63, init_cutoff_hz=300.0,  sample_rate=self.fs0//16, per_channel=False)

        pad = kernel_size // 2 if padding == "same" else 0
        self.act = nn.LeakyReLU(leak, inplace=True)

        # 첫 conv: 1 -> out_ch, 이후 conv: out_ch -> out_ch
        convs = []
        for i in range(4):
            in_ch = out_ch
            convs.append(nn.Conv1d(in_ch, out_ch, kernel_size, stride=1, padding=pad, bias=True))
        self.convs = nn.ModuleList(convs)
        self.map = MapTo10_F64_T100_CNN2D(
            hidden=10,      # internal conv2d hidden
            leak=leak,
        )

    def forward(self, x):
        """
        x: (B,4,48000)
        return: (B,4,1500,64)
        """
        B, C, T = x.shape
        
        # (B,4,48000) -> (B*4,48000,1)
        h = x.reshape(B * C, 1, T)

        h = self.gammatone(h)  # (B*4, C, T)
        h = self.aa0(h)       # (B*4, C, T)
        h = h[..., ::2]

        for i, conv in enumerate(self.convs):
            h = conv(h)        # (B*4, 64, L)
            h = self.act(h)
            if i == 0:
                h = self.aa1(h)
            elif i == 1:
                h = self.aa2(h)
            elif i == 2:
                h = self.aa3(h)
            elif i == 3:
                h = self.aa4(h)
            h = h[..., ::2]    # decimate (time) by 2 torch.Size([128, 64, 1500])

        # L: 48000 -> 24000 -> 12000 -> 6000 -> 3000 -> 1500
        # (B*4,64,1500) -> (B,4,1500,64)
        h = h.reshape(B, C, h.shape[1], h.shape[2]) #(B,4,64,1500)
        out = self.map(h) 
        out = out.permute(0,1,3,2).contiguous()
        return out

    
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResnetConformer_seddoa_nopool_2023(nn.Module):
    def __init__(self, in_channel, in_dim, out_dim):
        super().__init__()
        self.resnet = resnet18_nopool(in_channel=in_channel)
        embedding_dim = in_dim // 32 * 256
        encoder_dim = 256
        self.input_projection = nn.Sequential(
            nn.Linear(embedding_dim, encoder_dim),
            nn.Dropout(p=0.05),
        )
        num_layers = 8
        self.frontend = DownConvDecimate5x()
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
        N = T * L

        x = x.reshape(B,M,N).contiguous() # (B, 4, 48000)
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