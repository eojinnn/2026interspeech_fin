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
    def __init__(
        self,
        sr=24000,
        n_bins=64,
        n_lags=64,
        hop_length=480,
        win_length=960,
        n_fft=1024,
        scaler_path=None,
        gate_hidden=16,
        eps=1e-8,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_lags = n_lags
        self.n_freqs = n_fft // 2 + 1
        self.eps = eps

        self.init_slope = 3.0
        self.init_transition_ratio = 0.5

        # Mel filterbank
        mel_filters = torchaudio.functional.melscale_fbanks(
            n_freqs=self.n_freqs,
            n_mels=n_bins,
            f_min=0.0,
            f_max=sr / 2,
            sample_rate=sr,
            norm="slaney",
        )
        self.register_buffer("mel_filters", mel_filters)
        self.register_buffer("window", torch.hann_window(win_length))

        # Prior bias (log-weight space)
        # freq_axis: low -> 0, high -> 1
        freq_axis = torch.linspace(0.0, 1.0, steps=self.n_freqs)
        prior_curve = torch.tanh(self.init_slope * (freq_axis - self.init_transition_ratio))

        # shared importance prior는 neutral(0)로 시작
        self.register_buffer("shared_prior_log", torch.zeros(self.n_freqs))

        self.register_buffer("mel_prior_base", 0.35 * prior_curve)
        self.register_buffer("gcc_prior_base", -0.35 * prior_curve)

        self.mel_prior_delta = nn.Parameter(torch.zeros_like(prior_curve))
        self.gcc_prior_delta = nn.Parameter(torch.zeros_like(prior_curve))
        self.mel_prior_delta.requires_grad_(False)
        self.gcc_prior_delta.requires_grad_(False)

        self.prior_delta_limit = 0.15        # Summary network: [log_power, coherence, velocity] -> hidden
        self.summary_mlp = nn.Sequential(
            nn.Linear(3, gate_hidden),
            nn.Tanh(),
            nn.Linear(gate_hidden, gate_hidden),
            nn.Tanh(),
        )

        # Shared GRU-style gate
        self.shared_z = nn.Linear(gate_hidden, 1)
        self.shared_r = nn.Linear(gate_hidden, 1)
        self.shared_h = nn.Linear(gate_hidden + 1, 1)

        # Mel residual GRU-style gate
        self.mel_z = nn.Linear(gate_hidden, 1)
        self.mel_r = nn.Linear(gate_hidden, 1)
        self.mel_h = nn.Linear(gate_hidden + 1, 1)

        # GCC residual GRU-style gate
        self.gcc_z = nn.Linear(gate_hidden, 1)
        self.gcc_r = nn.Linear(gate_hidden, 1)
        self.gcc_h = nn.Linear(gate_hidden + 1, 1)

        # branch-wise post scaling
        self.mel_post_scale = nn.Parameter(torch.tensor(1.0))
        self.gcc_post_scale = nn.Parameter(torch.tensor(1.0))

        # log-weight scale / clamp range
        self.shared_scale = 0.20
        self.residual_scale = 0.12
        self.max_log_weight = 0.40   #exp(±0.4)≈[0.67,1.49]

        self.scaler = GlobalScalar(scaler_path)

        self._init_gate_params()

    def _init_gate_params(self):
        # 시작 시 baseline에 가깝게: update gate는 작게, candidate는 0 근처
        gate_layers = [
            self.shared_z, self.shared_r, self.shared_h,
            self.mel_z, self.mel_r, self.mel_h,
            self.gcc_z, self.gcc_r, self.gcc_h,
        ]
        for layer in gate_layers:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

        # update gate bias를 음수로 두어 초반엔 prior 유지
        self.shared_z.bias.data.fill_(-2.0)
        self.mel_z.bias.data.fill_(-2.0)
        self.gcc_z.bias.data.fill_(-2.0)

        # reset gate는 중립(0 -> sigmoid=0.5)
        self.shared_r.bias.data.fill_(0.0)
        self.mel_r.bias.data.fill_(0.0)
        self.gcc_r.bias.data.fill_(0.0)

    def set_prior_trainable(self, flag=True):
        self.mel_prior_delta.requires_grad_(flag)
        self.gcc_prior_delta.requires_grad_(flag)

    def get_current_prior_logs(self):
        mel_prior = self.mel_prior_base + self.prior_delta_limit * torch.tanh(self.mel_prior_delta)
        gcc_prior = self.gcc_prior_base + self.prior_delta_limit * torch.tanh(self.gcc_prior_delta)
        return mel_prior, gcc_prior

    def _standardize_freqwise(self, x):
        # x: [B, Freq]
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-5)
        return (x - mean) / std

    def _branch_norm(self, x):
        # x: [B, C, Feat, Frames]
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1e-5)
        return (x - mean) / std

    def _gru_style_update(self, feat, prior, z_layer, r_layer, h_layer):
        """
        feat  : [B, Freq, H]
        prior : [B, Freq]
        """
        z = torch.sigmoid(z_layer(feat)).squeeze(-1)   # [B, Freq]
        r = torch.sigmoid(r_layer(feat)).squeeze(-1)   # [B, Freq]

        cand_in = torch.cat([feat, (r * prior).unsqueeze(-1)], dim=-1)
        cand = torch.tanh(h_layer(cand_in)).squeeze(-1)  # [B, Freq]

        out = (1.0 - z) * prior + z * cand
        return out

    def _compute_gate_summary(self, stft, mag):
        """
        stft: [B, M, Freq, Frames] complex
        mag : [B, M, Freq, Frames] real
        """
        # 1) spectral summary: mean log-power
        power = mag.pow(2)
        log_power = torch.log1p(power.mean(dim=(1, 3)))   # [B, Freq]
        delta_power = torch.abs(power[:, :, :, 1:] - power[:, :, :, :-1]) # [B, M, Freq, Frames-1]
        
        # 2) spatial summary: mean coherence across mic pairs
        B, M, _, _ = stft.shape
        coh_list = []
        for m1 in range(M):
            for m2 in range(m1 + 1, M):
                X1 = stft[:, m1, :, :]   # [B, Freq, Frames]
                X2 = stft[:, m2, :, :]

                cross_mean = (X1 * torch.conj(X2)).mean(dim=-1)  # [B, Freq]
                psd1 = (X1.abs().pow(2)).mean(dim=-1)
                psd2 = (X2.abs().pow(2)).mean(dim=-1)

                coh = cross_mean.abs() / (torch.sqrt(psd1 * psd2) + self.eps)
                coh = coh.clamp(0.0, 1.0)
                coh_list.append(coh)

        spatial_conf = torch.stack(coh_list, dim=1).mean(dim=1)  # [B, Freq]
        velocity_summary = torch.log1p(delta_power.amax(dim=3).mean(dim=1)) # [B, Freq]

        # normalize summaries per sample
        log_power = self._standardize_freqwise(log_power)
        spatial_conf = self._standardize_freqwise(spatial_conf)
        velocity_summary = self._standardize_freqwise(velocity_summary)

        summary = torch.stack([log_power, spatial_conf,velocity_summary], dim=-1)  # [B, Freq, 3]
        feat = self.summary_mlp(summary)  # [B, Freq, H]
        return feat

    def get_weights(self, stft, mag):
        """
        Return:
            mel_weight: [B, Freq]
            gcc_weight: [B, Freq]
        """
        B = stft.shape[0]
        feat = self._compute_gate_summary(stft, mag)  # [B, Freq, H]

        shared_prior = self.shared_prior_log.unsqueeze(0).expand(B, -1)

        mel_prior_1d, gcc_prior_1d = self.get_current_prior_logs()
        mel_prior = mel_prior_1d.unsqueeze(0).expand(B, -1)
        gcc_prior = gcc_prior_1d.unsqueeze(0).expand(B, -1)

        shared_log = self._gru_style_update(
            feat, shared_prior, self.shared_z, self.shared_r, self.shared_h
        )
        mel_res_log = self._gru_style_update(
            feat, mel_prior, self.mel_z, self.mel_r, self.mel_h
        )
        gcc_res_log = self._gru_style_update(
            feat, gcc_prior, self.gcc_z, self.gcc_r, self.gcc_h
        )

        mel_log = self.shared_scale * shared_log + self.residual_scale * mel_res_log
        gcc_log = self.shared_scale * shared_log + self.residual_scale * gcc_res_log

        mel_log = torch.clamp(mel_log, -self.max_log_weight, self.max_log_weight)
        gcc_log = torch.clamp(gcc_log, -self.max_log_weight, self.max_log_weight)

        mel_weight = torch.exp(mel_log)   # [B, Freq]
        gcc_weight = torch.exp(gcc_log)   # [B, Freq]

        return mel_weight, gcc_weight

    def forward(self, wav, return_weights=False):
        """
        wav: [B, M, F, T]
        """
        B, M, n_frames_in, T = wav.shape
        wav = wav.reshape(B * M, n_frames_in * T)

        # 1) shared STFT
        stft = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        )
        # [B*M, Freq, Frames] -> [B, M, Freq, Frames]
        stft = stft.reshape(B, M, stft.shape[1], stft.shape[2])

        if stft.shape[-1] > n_frames_in:
            stft = stft[:, :, :, :n_frames_in]

        mag = torch.abs(stft)

        # 2) dynamic GRU-style weights
        mel_weight, gcc_weight = self.get_weights(stft, mag)
        mel_weight = mel_weight.to(stft.device)
        gcc_weight = gcc_weight.to(stft.device)

        # 3) Mel branch
        mel_w = mel_weight.unsqueeze(1).unsqueeze(-1)   # [B, 1, Freq, 1]
        power = (mag ** 2) * mel_w
        mel = torch.matmul(
            power.transpose(-1, -2), self.mel_filters
        ).transpose(-1, -2)   # [B, M, n_mels, Frames]

        mel = 10.0 * torch.log10(mel.clamp_min(1e-10))
        max_val = mel.amax(dim=(-2, -1), keepdim=True)
        mel = torch.maximum(mel, max_val - 80.0)

        # branch-wise normalization to preserve relative usefulness
        # mel = self.mel_post_scale * self._branch_norm(mel)
        mel = self.mel_post_scale * mel

        # 4) GCC-PHAT branch
        cc_list = []
        for m1 in range(M):
            for m2 in range(m1 + 1, M):
                X1 = stft[:, m1, :, :]
                X2 = stft[:, m2, :, :]

                R = X1 * torch.conj(X2)
                R_phat = R / (torch.abs(R) + 1e-8)

                gcc_w = gcc_weight.unsqueeze(-1)   # [B, Freq, 1]
                R_weighted = R_phat * gcc_w

                cc = torch.fft.irfft(R_weighted, n=self.n_fft, dim=1)
                cc = torch.fft.fftshift(cc, dim=1)

                center = self.n_fft // 2
                start = center - self.n_lags // 2
                end = start + self.n_lags
                cc_cropped = cc[:, start:end, :]   # [B, n_lags, Frames]
                cc_list.append(cc_cropped)

        gcc = torch.stack(cc_list, dim=1)   # [B, num_pairs, n_lags, Frames]
        # gcc = self.gcc_post_scale * self._branch_norm(gcc)
        gcc = self.gcc_post_scale * gcc

        # 5) concat
        out = torch.cat([mel, gcc], dim=1)   # [B, M+num_pairs, Feat, Frames]
        out = out.permute(0, 1, 3, 2)        # [B, C, Time, Feat]
        out = self.scaler(out)

        if return_weights:
            mel_prior_1d, gcc_prior_1d = self.get_current_prior_logs()
            return out, {
                "mel_weight": mel_weight,
                "gcc_weight": gcc_weight,
                "mel_prior_log": mel_prior_1d.detach(),
                "gcc_prior_log": gcc_prior_1d.detach(),
                "mel_prior_delta": self.mel_prior_delta.detach(),
                "gcc_prior_delta": self.gcc_prior_delta.detach(),
            }
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

    def set_frontend_prior_trainable(self, flag=True):
        self.frontend.set_prior_trainable(flag)

    def forward(self, x, return_frontend_weights=False):
        B, M, T, L = x.shape
        x = self.frontend(x, return_weights=return_frontend_weights)
        frontend_info = None
        if return_frontend_weights:
            x, frontend_info = x
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
        if return_frontend_weights:
            return pred, frontend_info
        # pdb.set_trace()
        return pred