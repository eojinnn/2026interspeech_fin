import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import resnet18_nopool
from .conformer import ConformerBlock

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return x


# ---------------------------------------------------------
# 💡 단순화 및 차원 오류를 수정한 conv_encoder
# ---------------------------------------------------------
class conv_encoder(torch.nn.Module):
    def __init__(self, in_channels, params):
        super().__init__()
        self.params = params
        self.t_pooling_loc = params.get("t_pooling_loc", "front")
        assert len(params['f_pool_size']) > 0

        self.conv_block_list = nn.ModuleList()

        for conv_cnt in range(len(params['f_pool_size'])):
            self.conv_block_list.append(nn.Sequential(
                ConvBlock(
                    in_channels=params['nb_cnn2d_filt'] if conv_cnt else in_channels,
                    out_channels=params['nb_cnn2d_filt']
                ),
                nn.MaxPool2d((
                    params['f_pool_size'][conv_cnt], # H: 주파수 차원 풀링
                    params['t_pool_size'][conv_cnt] if self.t_pooling_loc == 'front' else 1 # W: 시간 차원 풀링
                )),
                nn.Dropout2d(p=params.get('dropout_rate', 0.1)),
            ))

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for block in self.conv_block_list:
            x = block(x)  # out: B, C, F, T
        return x
    

# -------------------------
# GCC-PHAT (unchanged)
# -------------------------
class GCC_PHAT(nn.Module):
    def __init__(self, n_lags=64):
        super().__init__()
        self.n_lags = n_lags

    def forward(self, x):
        # x shape: [B, M(4), T(500), L(480)]
        B, M, T, L = x.shape
        X = torch.fft.rfft(x, dim=-1)

        cc_list = []
        for m1 in range(M):
            for m2 in range(m1 + 1, M):
                X1 = X[:, m1, :, :]
                X2 = X[:, m2, :, :]
                R = X1 * torch.conj(X2)
                R = R / (torch.abs(R) + 1e-8)
                cc = torch.fft.irfft(R, dim=-1)
                cc = torch.fft.fftshift(cc, dim=-1)

                center = L // 2
                start = center - self.n_lags // 2
                end = start + self.n_lags

                # 안전 슬라이싱 (혹시 L이 달라져도 터지지 않게)
                start = max(0, start)
                end = min(L, end)
                cc = cc[:, :, start:end]  # [B, T, <=n_lags]
                if cc.size(-1) < self.n_lags:
                    cc = F.pad(cc, (0, self.n_lags - cc.size(-1)))
                elif cc.size(-1) > self.n_lags:
                    cc = cc[:, :, :self.n_lags]

                cc_list.append(cc)

        out = torch.stack(cc_list, dim=1)  # [B, 6, T, n_lags]
        return out


# -------------------------
# ERB utilities (for gammatone center freqs)
# -------------------------
def hz_to_erb_number(f_hz: np.ndarray) -> np.ndarray:
    return 21.4 * np.log10(1.0 + 0.00437 * f_hz)

def erb_number_to_hz(erb: np.ndarray) -> np.ndarray:
    return (10 ** (erb / 21.4) - 1.0) / 0.00437

def erb_bandwidth(f_hz: np.ndarray) -> np.ndarray:
    return 24.7 * (4.37 * f_hz / 1000.0 + 1.0)


# -------------------------
# Fixed Gammatone FIR bank (Conv1d)
# -------------------------
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
        kernel_size: int = 1024,
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


# -------------------------
# Learnable IHC low-pass (Depthwise SincConv)
# -------------------------
class DepthwiseSincLowPass(nn.Module):
    """
    Per-channel learnable low-pass using windowed sinc.
    Input:  (B, C, N)
    Output: (B, C, N)
    """
    def __init__(
        self,
        num_channels: int,
        sample_rate: int = 24000,
        kernel_size: int = 129,
        min_cutoff_hz: float = 20.0,
        max_cutoff_hz: float = 5000.0,
        init_cutoff_hz: torch.Tensor | None = None,
    ):
        super().__init__()
        self.C = int(num_channels)
        self.fs = int(sample_rate)
        self.kernel_size = int(kernel_size)
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1

        self.min_cut = float(min_cutoff_hz)
        self.max_cut = float(min(max_cutoff_hz, self.fs / 2.0 - 100.0))

        t = torch.arange(-(self.kernel_size // 2), self.kernel_size // 2 + 1, dtype=torch.float32) / self.fs
        self.register_buffer("t", t.view(1, -1))  # (1,K)
        self.register_buffer("window", torch.hamming_window(self.kernel_size, periodic=False).view(1, -1))  # (1,K)

        if init_cutoff_hz is None:
            init = torch.full((self.C, 1), 1000.0, dtype=torch.float32)
        else:
            init = init_cutoff_hz.view(self.C, 1).to(dtype=torch.float32)

        init = torch.clamp(init, self.min_cut, self.max_cut)

        # cutoff = min + (max-min)*sigmoid(p)
        frac = (init - self.min_cut) / (self.max_cut - self.min_cut + 1e-12)
        frac = torch.clamp(frac, 1e-4, 1.0 - 1e-4)
        p0 = torch.log(frac / (1.0 - frac))
        self.p_cut = nn.Parameter(p0)

    def cutoff_hz(self) -> torch.Tensor:
        frac = torch.sigmoid(self.p_cut)
        return self.min_cut + (self.max_cut - self.min_cut) * frac  # (C,1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,N)
        cut = self.cutoff_hz().to(dtype=x.dtype, device=x.device)  # (C,1)
        t = self.t.to(dtype=x.dtype, device=x.device)              # (1,K)
        win = self.window.to(dtype=x.dtype, device=x.device)       # (1,K)

        arg = 2.0 * cut * t        # (C,K)
        h = (2.0 * cut) * torch.sinc(arg)  # (C,K)
        h = h * win
        h = h / (h.sum(dim=-1, keepdim=True) + 1e-8)  # DC gain ~ 1

        weight = h.view(self.C, 1, self.kernel_size)
        pad = self.kernel_size // 2
        return F.conv1d(x, weight, stride=3, padding=pad, groups=self.C)


# -------------------------
# Cochlea + IHC frontend (continuous 240000 processing)
# -------------------------
class CochleaIHCFrontend(nn.Module):
    """
    Input:  audio frames (B,M,T,L)
    Steps:
      - concat frames -> (B*M, 1, N=T*L)
      - fixed gammatone FIR -> (B*M, C, N)
      - rectify abs
      - learnable IHC low-pass (depthwise) -> (B*M, C, N)
      - (optional) log compression
      - pool back to frames -> (B,M,T,C)
    """
    def __init__(
        self,
        sample_rate=24000,
        num_filters=64,
        gammatone_kernel=1024,
        ihc_kernel=129,
        ihc_min_cut=20.0,
        ihc_max_cut=5000.0,
        use_log=True,
    ):
        super().__init__()
        self.C = num_filters
        self.use_log = use_log

        self.gammatone = FixedGammatoneFIR(
            sample_rate=sample_rate,
            num_filters=num_filters,
            min_freq=50.0,
            max_freq=sample_rate / 2.0,
            kernel_size=gammatone_kernel,
            order_n=4,
            scale_b=1.019,
        )

        init_cut = 0.25 * self.gammatone.center_freqs_hz  # heuristic init
        self.ihc = DepthwiseSincLowPass(
            num_channels=num_filters,
            sample_rate=sample_rate,
            kernel_size=ihc_kernel,
            min_cutoff_hz=ihc_min_cut,
            max_cutoff_hz=ihc_max_cut,
            init_cutoff_hz=init_cut,
        )

        encoder_params = {
            "t_pooling_loc": "front",
            "t_pool_size": [4,4],   # 주파수(혹은 샘플 L) 축을 480 -> 1 로 줄임
            "f_pool_size": [1,1],   # T 차원은 GCC-PHAT와 결합하기 위해 유지
            "nb_cnn2d_filt": 1, 
            "dropout_rate": 0.05
        }
        self.conv_pooler = conv_encoder(in_channels=1, params=encoder_params)
        self.fusion_conv = nn.Conv1d(in_channels=self.C * 2, out_channels=self.C, kernel_size=1)
        nn.init.kaiming_uniform_(self.fusion_conv.weight.data, nonlinearity='relu')

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: (B,M,T,L) -> feat: (B,M,T,C)
        B, M, T, L = audio.shape
        N = T * L

        x = audio.reshape(B * M, 1, N).contiguous()

        # 안정 디버깅 시에는 FFT/필터 구간만 fp32로 고정해도 좋음
        # x = x.float()

        sub = self.gammatone(x)       # (B*M, C, N)
        sub = torch.abs(sub)          # rectify
        env = self.ihc(sub)           # (B*M, C, N) low-pass smoothing

        if self.use_log:
            env = torch.log1p(env.clamp_min(0.0))

        peak_env = F.max_pool1d(env, kernel_size=160, stride=160) # [B*M, 64, 500]
        # frame pooling: (B*M,C,T,L) -> max over L -> (B*M,C,T)
        # env = env.view(B * M, self.C, T, L).amax(dim=-1)
        env = F.max_pool1d(env, kernel_size=10, stride=10)
        env = env.view(B * M, 1, self.C, -1)
        env = self.conv_pooler(env)
        env = env.squeeze(1)  # (B*M, C, T)

        merged_env = torch.cat([peak_env, env], dim=1) # [B*M, 128, 500]
        out_env = self.fusion_conv(merged_env) # [B*M, 64, 500]
        # -> (B,M,T,C)
        feat = out_env.view(B, M, self.C, T).permute(0, 1, 3, 2).contiguous()
        return feat


# -------------------------
# Final model (complete)
# -------------------------
class ResnetConformer_seddoa_nopool_2023(nn.Module):
    def __init__(
        self,
        in_channel,  # not used directly now; kept for compatibility
        in_dim,
        out_dim,
        fs=24000,
        sig_len=480,
        num_channels=64,
        normalize_output=False,
        pool_len=5,
        gammatone_kernel=1024,
        ihc_kernel=129,
    ):
        super().__init__()

        # 1) Frontend: cochlea + IHC
        self.frontend = CochleaIHCFrontend(
            sample_rate=fs,
            num_filters=num_channels,
            gammatone_kernel=gammatone_kernel,
            ihc_kernel=ihc_kernel,
            ihc_min_cut=20.0,
            ihc_max_cut=5000.0,
            use_log=True,
        )

        # 2) GCC-PHAT
        self.gcc = GCC_PHAT(n_lags=64)

        # 3) ResNet input: Mics(4) + GCC pairs(6) = 10 channels, image-like (T x 64)
        self.resnet = resnet18_nopool(in_channel=4 + 6)

        # 4) Conformer + outputs (kept from your original)
        embedding_dim = in_dim // 32 * 256
        encoder_dim = 256

        self.input_projection = nn.Sequential(
            nn.Linear(embedding_dim, encoder_dim),
            nn.Dropout(p=0.05),
        )

        self.conformer_layers = nn.ModuleList([
            ConformerBlock(
                dim=encoder_dim,
                dim_head=32,
                heads=8,
                ff_mult=2,
                conv_expansion_factor=2,
                conv_kernel_size=7,
                attn_dropout=0.1,
                ff_dropout=0.1,
                conv_dropout=0.1
            )
            for _ in range(8)
        ])

        self.t_pooling = nn.MaxPool1d(kernel_size=pool_len)

        self.sed_out_layer = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim), nn.LeakyReLU(),
            nn.Linear(encoder_dim, 13), nn.Sigmoid()
        )

        self.out_layer = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim), nn.LeakyReLU(),
            nn.Linear(encoder_dim, out_dim), nn.Tanh()
        )

    def forward(self, audio):
        """
        audio: [B, M(4), T(500), L(480)]
        return: [B, T', 13+out_dim]  (T' depends on pooling)
        """
        B, M, T, L = audio.shape

        # GCC path
        cc = self.gcc(audio)  # [B, 6, T, 64]

        # Frontend path (continuous 240000 processing inside)
        x_feat = self.frontend(audio)  # [B, 4, T, 64]

        # Fusion
        x = torch.cat([x_feat, cc], dim=1)  # [B, 10, T, 64]
        print(f"After fusion: {x.shape}")  # 디버깅용 출력
        # Downstream
        conv_outputs = self.resnet(x)  # expected [B, C, T_out, W]
        N, C, T_out, W = conv_outputs.shape
        conv_outputs = conv_outputs.permute(0, 2, 1, 3).reshape(N, T_out, C * W)

        conformer_outputs = self.input_projection(conv_outputs)
        for layer in self.conformer_layers:
            conformer_outputs = layer(conformer_outputs)

        outputs = conformer_outputs.permute(0, 2, 1)  # [B, enc, T_out]
        outputs = self.t_pooling(outputs)             # [B, enc, T']
        outputs = outputs.permute(0, 2, 1)            # [B, T', enc]

        sed = self.sed_out_layer(outputs)
        doa = self.out_layer(outputs)
        return torch.cat((sed, doa), dim=-1)
