import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import resnet18_nopool
from .conformer import ConformerBlock


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
        return F.conv1d(x, weight, padding=pad, groups=self.C)


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
    def __init__(self, sample_rate=24000, num_filters=64, gammatone_kernel=1024, ihc_kernel=129, ihc_min_cut=20.0, ihc_max_cut=5000.0, eps=1e-8,
        itd_max_hz=1500.0, itd_max_bands=16, itd_n_lags=64, itd_use_norm=True,
        ):
        super().__init__()
        self.C = num_filters
        self.eps = eps
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
        m1 = torch.tensor([0, 0, 0, 1, 1, 2], dtype=torch.long)
        m2 = torch.tensor([1, 2, 3, 2, 3, 3], dtype=torch.long)
        self.register_buffer("pair_m1", m1)
        self.register_buffer("pair_m2", m2)

        self.itd_n_lags = int(itd_n_lags)
        self.itd_use_norm = bool(itd_use_norm)
        cf = self.gammatone.center_freqs_hz  # (C,)
        low_idx = torch.nonzero(cf <= float(itd_max_hz), as_tuple=False).squeeze(-1)
        if low_idx.numel() == 0:
            # 안전장치: 최소 1개는 선택
            low_idx = torch.tensor([0], dtype=torch.long)
        if low_idx.numel() > int(itd_max_bands):
            # 균등 subsample
            sel = torch.linspace(0, low_idx.numel() - 1, int(itd_max_bands)).round().long()
            low_idx = low_idx[sel]
        self.register_buffer("itd_low_idx", low_idx)  # (K,)

        # self.ild_mix = nn.Conv2d(6, 6, kernel_size=1, bias=True)
        # self.itd_mix = nn.Conv2d(6, 6, kernel_size=1, bias=True)
    
        self.gn = nn.GroupNorm(num_groups=13, num_channels=13)  # (B,12,T,W)
        self.mix = nn.Conv2d(13, 13, kernel_size=1, bias=True)
        with torch.no_grad():
            self.mix.weight.zero_()
            for i in range(13):
                self.mix.weight[i, i, 0, 0] = 1.0
            self.mix.bias.zero_()

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: (B,M,T,L) -> feat: (B,M,T,C)
        B, M, T, L = audio.shape
        N = T * L

        x = audio.reshape(B * M, 1, N).contiguous()

        # 안정 디버깅 시에는 FFT/필터 구간만 fp32로 고정해도 좋음
        # x = x.float()
        sub_raw = self.gammatone(x)       # (B*M, C, N)
        sub = torch.abs(sub_raw)          # rectify
        env = self.ihc(sub)           # (B*M, C, N) low-pass smoothing
        env_frames = env.view(B, M, self.C, T, L)             # (B,M,C,T,L)

        #mono
        env_mono = env_frames.mean(dim=1)           # (B,C,T,L)
        env_mono = env_mono.mean(dim=-1)            # (B,C,T)
        env_mono = env_mono.permute(0, 2, 1).contiguous()  # (B,T,C)
        env_mono = env_mono.unsqueeze(1)     

        #ild
        E = (env_frames ** 2).sum(dim=-1).permute(0, 1, 3, 2).contiguous()  # (B,M,T,C)

        ild_list = []
        for p in range(6):
            m1_idx = int(self.pair_m1[p].item())
            m2_idx = int(self.pair_m2[p].item())

            E_m1 = E[:, m1_idx]  # (B,T,C)
            E_m2 = E[:, m2_idx]  # (B,T,C)

            ild_p = 10.0 * torch.log10((E_m2 + self.eps) / (E_m1 + self.eps))  # (B,T,C)
            ild_list.append(ild_p)
        ild = torch.stack(ild_list, dim=1)
        
        #itd
        sub_frames = sub_raw.view(B, M, self.C, T, L)  # (B,M,C,T,L)

        itd_pair_list = []
        half = self.itd_n_lags // 2

        for p in range(6):
            m1_idx = int(self.pair_m1[p].item())
            m2_idx = int(self.pair_m2[p].item())

            # (B,K,T,L) directly (no sub_low)
            s1_p = sub_frames[:, m1_idx, self.itd_low_idx, :, :]
            s2_p = sub_frames[:, m2_idx, self.itd_low_idx, :, :]

            # DC remove per frame
            s1_p = s1_p - s1_p.mean(dim=-1, keepdim=True)
            s2_p = s2_p - s2_p.mean(dim=-1, keepdim=True)

            X1_p = torch.fft.rfft(s1_p, dim=-1)
            X2_p = torch.fft.rfft(s2_p, dim=-1)
            R_p = X1_p * torch.conj(X2_p)

            if self.itd_use_norm:
                denom = (torch.abs(X1_p) * torch.abs(X2_p)).clamp_min(self.eps)
                R_p = R_p / denom

            cc_p = torch.fft.irfft(R_p, n=L, dim=-1)  # (B,K,T,L)

            # window around 0-lag without fftshift: concat [neg | pos]
            neg = cc_p[..., -half:]                          # (B,K,T,half)
            pos = cc_p[..., :self.itd_n_lags - half]         # (B,K,T,n_lags-half)
            cc_cropped = torch.cat([neg, pos], dim=-1)       # (B,K,T,n_lags)

            # band weighting
            band_energy = s1_p.pow(2).mean(dim=-1) + s2_p.pow(2).mean(dim=-1)  # (B,K,T)
            weight = torch.softmax(band_energy, dim=1).unsqueeze(-1)          # (B,K,T,1)

            itd_cc_p = (cc_cropped * weight).sum(dim=1)       # (B,T,n_lags)
            itd_pair_list.append(itd_cc_p)

        itd_cc = torch.stack(itd_pair_list, dim=1)            # (B,6,T,n_lags)

        ild = ild.clamp(-30.0,30.0)/30.0
        x = torch.cat([env_mono, ild, itd_cc], dim=1)      # (B,12,T,64)
        x = self.gn(x)                           # GroupNorm(12,12)
        feat = self.mix(x)                          # Conv2d(12,12,1)
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
        )

        # 3) ResNet input: Mics(4) + GCC pairs(6) = 10 channels, image-like (T x 64)
        self.resnet = resnet18_nopool(in_channel=13)

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
            nn.Linear(encoder_dim, encoder_dim), 
            nn.LeakyReLU(),
            nn.Linear(encoder_dim, 13), 
            # nn.Sigmoid()
        )

        self.out_layer = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim), 
            nn.LeakyReLU(),
            nn.Linear(encoder_dim, out_dim), 
            nn.Tanh()
        )

    def forward(self, audio):
        """
        audio: [B, M(4), T(500), L(480)]
        return: [B, T', 13+out_dim]  (T' depends on pooling)
        """
        B, M, T, L = audio.shape

        # Frontend path (continuous 240000 processing inside)
        x_feat = self.frontend(audio)  # [B, 12, T, 64]

        # Downstream
        conv_outputs = self.resnet(x_feat)  # expected [B, C, T_out, W]
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
